"""Build a detailed virtual try-on prompt from a fit-calculation dict.

Two modes:
    build_fit_prompt_llm(fit_result, garment_image_path=None)
        -> asks Groq to write a realistic ~500-word prompt from the fit data
           and style instructions in the meta prompt.
           More natural, style-aware, handles edge cases better. Needs API key.

    build_fit_prompt(fit_result)
        -> deterministic rule-based version. Offline fallback, no API call.

Both produce a prompt suitable for Gemini 2.5 Flash Image (Nano Banana).

All band thresholds below assume catalog units of cm.
"""


def build_fit_prompt_llm(fit_result: dict, garment_image_path: str | None = None) -> str:
    """Ask Groq to write the final try-on prompt from fit + style instructions.

    Groq is used only for prompt-writing. Gemini Nano Banana receives the
    resulting prompt later for actual image generation.

    Groq text generation is text-only in this project, so the garment image
    path is accepted for API compatibility and ignored here.
    """
    from .groq_client import generate_text

    meta_prompt = build_meta_prompt(fit_result)
    return generate_text(meta_prompt)


def build_meta_prompt(fit_result: dict) -> str:
    name = fit_result.get("garment_name", "the garment")
    gid = fit_result.get("garment_id", "?")
    size = fit_result.get("garment_size", "unspecified")
    fit_style = fit_result.get("garment_fit_style", "standard")
    units = fit_result.get("units", "cm")
    category = fit_result.get("category", "top")
    p = fit_result["person_dimensions"]
    g = fit_result["garment_dimensions"]
    d = fit_result["deltas"]

    rows = []
    for key in d:
        person_v = p.get(key, "?")
        garment_v = g.get(key, "?")
        rows.append(
            f"- {key.replace('_', ' ').capitalize():<10} person {person_v} | garment {garment_v} | delta {d[key]:+}"
        )
    table = "\n".join(rows)

    upper_bands = """
CHEST (circumference, cm):
  0 to ±2.5    -> true skim fit, fabric just kisses the body, no tension, no slack.
  -2.5 to -7.5 -> snug. Fabric in light contact, follows the bust/ribcage outline, no visible stretch lines yet, prints undistorted.
  -7.5 to -15  -> clearly tight. Faint horizontal tension lines radiating from the underarm seams, body contour reads through the fabric, side seams begin to bow outward.
  -15 to -25   -> heavily strained, bodycon-style. Pronounced tension lines, ribcage and bust outline visible through cloth, side seams under obvious load, any buttons gape, prints stretched and distorted.
  -25 to -38   -> the fabric is at near-maximum stretch. Woven texture visibly thinned, color slightly washed where stretched, garment unmistakably the wrong size but still wearable on a stretch fabric.
  beyond -38   -> describe as "fabric at maximum possible stretch, garment clearly the wrong size for this wearer." Do NOT describe tearing, ripping, popping seams, or impossibility. The image must remain photographically plausible — a too-small garment, not a destroyed one.
  +2.5 to +7.5 -> relaxed, small air gap, soft straight drape from the shoulder.
  +7.5 to +15  -> loose, gentle vertical folds from the shoulder, fabric clearly off the body.
  +15 to +30   -> oversized, deep folds, tent-leaning silhouette.
  beyond +30   -> very oversized, drowning, fabric pools and pleats heavily.

SHOULDER (seam-to-seam width, cm):
  0 to ±2.5    -> seam sits exactly on the shoulder bone, clean tailored line.
  -2.5 to -7.5 -> seam inboard of the shoulder bone, mild upward pull on the sleeve, faint strain across the upper back.
  -7.5 to -15  -> seam well inside the shoulder, sleeves yanked upward, armhole high and constricted, visible strain to the neckline.
  beyond -15   -> garment fits like one or two sizes too small at the yoke, seam approaches the collarbone area, fabric at maximum lateral stretch — keep it plausible, do not depict ripping.
  +2.5 to +7.5 -> dropped seam, soft casual line, slight fabric softening at the seam.
  +7.5 to +15  -> clearly dropped, streetwear oversized look, fabric pools at the armhole.
  beyond +15   -> heavily dropped, seam down at the upper bicep, tent silhouette.

LENGTH (torso, cm):
  0 to ±2.5    -> hem at the natural waist, clean horizontal line.
  -2.5 to -7.5 -> short, hem above the natural waist, a narrow strip of midriff or waistband visible.
  -7.5 to -15  -> cropped, hem at the lower ribcage, clear midriff exposure.
  -15 to -25   -> heavily cropped, hem near the upper ribcage, large section of midriff exposed.
  beyond -25   -> extreme — hem at chest level, garment reads almost like a bandeau or under-bust crop. Keep it plausible, do not invent a different garment.
  +2.5 to +7.5 -> long, hem past the hip, slight pooling.
  +7.5 to +15  -> tunic length, hem at upper thigh, vertical folds.
  beyond +15   -> very long, hem to mid-thigh or knee, dress-like silhouette.

ARM / SLEEVE (sleeve length, cm):
  0 to ±2.5    -> sleeve hem at the intended landmark.
  -2.5 to -7.5 -> sleeve clearly short, visible exposure beyond the cuff.
  -7.5 to -15  -> very short, sleeve barely covers the upper arm/shoulder, looks like a cropped or cap sleeve regardless of the original sleeve type.
  beyond -15   -> sleeve barely exists, garment reads as nearly sleeveless on this wearer. Keep it plausible.
  +2.5 to +7.5 -> long, fabric bunches at the cuff.
  +7.5 to +15  -> very long, sleeve covers the wrist and partial back of hand.
  beyond +15   -> extreme, sleeve fully covers the hand or extends past the fingertips.
""".strip()

    lower_bands = """
WAIST / HIP (circumference, cm):
  0 to ±2.5    -> true fit at the waistband, garment sits without slipping or pinching.
  -2.5 to -7.5 -> snug, waistband presses lightly into the body, faint impression visible above the hem.
  -7.5 to -15  -> clearly tight, visible waistband impression, fabric strained around the seat and hips.
  beyond -15   -> heavily strained, waistband digs into the body, side seams under load, prints distorted. Keep render plausible, no tearing.
  +2.5 to +7.5 -> relaxed, waistband sits comfortably with no compression.
  +7.5 to +15  -> loose, waistband sags, gap visible at the back, garment hangs off the hip.
  beyond +15   -> oversized, waistband well off the body, garment looks like the wrong size — render belt-cinched if appropriate.

INSEAM (cm):
  0 to ±2.5    -> hem breaks correctly at the ankle.
  -2.5 to -7.5 -> short, hem rides above the ankle bone (ankle pant look) regardless of intended cut.
  -7.5 to -15  -> very short, hem near mid-calf, garment reads as cropped.
  beyond -15   -> extreme — hem near the knee or upper calf, garment looks like a different cut entirely; keep plausible.
  +2.5 to +7.5 -> long, fabric stacks at the ankle with one or two folds above the shoe.
  +7.5 to +15  -> very long, heavy fabric stacking covers the top of the shoe.
  beyond +15   -> extreme, hem drags or pools on the ground.

OUTSEAM (hip-to-ankle, cm): use the same banding as inseam.

THIGH (circumference, cm):
  0 to ±2.5    -> clean fit through the thigh, fabric follows the leg without strain or excess.
  -2.5 to -7.5 -> snug, fabric in contact with the thigh, light tension visible front and back.
  -7.5 to -15  -> tight, clear thigh contour reading through the fabric, horizontal tension lines on movement-prone areas.
  beyond -15   -> heavily strained at the thigh, prints distorted, side seams under load.
  +2.5 to +7.5 -> relaxed, slight air gap around the thigh.
  +7.5 to +15  -> loose, clear drape, baggy line from hip to knee.
  beyond +15   -> oversized, deep folds, wide-leg silhouette regardless of intended cut.
""".strip()

    bands = upper_bands if category == "top" else f"{upper_bands}\n\n{lower_bands}"

    body_focus = "upper-body clothing" if category == "top" else "lower-body clothing"

    return f"""\
You are a senior fashion-fit expert AND a prompt engineer for photorealistic image-generation models. Your job: given precise body-vs-garment measurements, write ONE detailed image-generation prompt that tells the model exactly how the garment should physically behave on this specific person.

INPUTS:
Garment: "{name}" (catalog ID {gid}), labeled size {size}, intended cut "{fit_style}", category "{category}".
Units: {units}.

Dimension table (person vs. garment vs. delta):
{table}

Delta convention: garment − person (in {units}).
Negative -> garment smaller than body at that landmark -> compression/stretch/exposure.
Positive -> garment larger than body -> drape/excess/bunching.

MAGNITUDE CALIBRATION — this is the most important section. Severity MUST scale with the size of the delta. Use these bands as the physical reference for your language. Do NOT exaggerate small deltas, and do NOT minimize large ones.

{bands}

INTENT MODULATION — interpret each band against the cut ("{fit_style}"):
- A small negative delta on a Bodycon/Slim cut is intentional and reads sleek and tailored.
- The same delta on a Relaxed/Oversized/Boxy cut reads as unintentionally tight and strained.
- A positive delta on an Oversized cut reads as the intended drape; on a Slim/Fitted cut it reads as the wrong size.
- Always describe what the wearer would actually look like, not just the abstract physics.

WHAT YOUR OUTPUT PROMPT MUST DO:
1. Open with a clear TASK line: render a photorealistic try-on of this specific person wearing this specific garment; preserve face, hair, skin tone, pose, background; only the {body_focus} changes.
2. Explicitly instruct the image model to preserve every visible design detail from the garment reference image it will receive separately: print/logo/color blocking, collar/waistband shape, sleeve/leg type, fabric weight, seams, cuffs, hems, and texture.
3. Go through EACH delta using the band that the actual numeric delta falls into above. Be concrete and physical. Do not blur band boundaries.
4. Add an OVERALL SILHOUETTE sentence synthesizing the combined impression and naming the dominant problem area (or confirming a clean fit).
5. Close with RENDERING instructions: photograph-grade realism, match lighting and shadow softness of the person reference, no painterly/cartoon/stylized effects, single final image only, no added accessories, no background changes, and the result must remain a physically plausible photo even when deltas are extreme.

LENGTH: 450-600 words. Cohesive paragraphs, not bullet lists (except the per-landmark fit analysis may use short bulleted lines if that reads cleaner).

OUTPUT: the prompt text ONLY. No preamble ("Here is the prompt:"), no markdown code fences, no trailing commentary. Start directly with "TASK:"."""


def build_fit_prompt(fit_result: dict) -> str:
    gid = fit_result.get("garment_id", "?")
    name = fit_result.get("garment_name", "the garment")
    size = fit_result.get("garment_size", "unspecified size")
    fit_style = fit_result.get("garment_fit_style", "standard cut")
    units = fit_result.get("units", "cm")
    category = fit_result.get("category", "top")
    person = fit_result["person_dimensions"]
    garment = fit_result["garment_dimensions"]
    deltas = fit_result["deltas"]

    fit_sections = "\n\n".join(
        _describe(dim, deltas[dim], person.get(dim), garment.get(dim), units)
        for dim in deltas
    )
    silhouette = _overall_silhouette(deltas, fit_style)

    body_focus = "upper-body" if category == "top" else "lower-body"
    subject_dims_text = ", ".join(
        f"{k.replace('_', ' ')} {v} {units}" for k, v in person.items()
    )
    garment_dims_text = ", ".join(
        f"{k.replace('_', ' ')} {v} {units}" for k, v in garment.items()
    )

    prompt = f"""\
TASK — Produce a single photorealistic virtual try-on image. The first reference image shows the person; the second shows the garment as a flat product photo. Render the person wearing that exact garment on the {body_focus}, with every visible design feature preserved. Keep the person's face, hair, skin tone, pose, proportions, and the original background completely unchanged. The only change to the scene is the {body_focus} clothing. Do not add accessories, crops, stylization, or extra subjects.

SUBJECT PROFILE — The wearer's measured {body_focus} dimensions are: {subject_dims_text}. These define the body landmarks the garment must respect. Do not alter the wearer's build; the body stays as shown, and the fabric must conform to it.

GARMENT PROFILE — The target piece is a "{name}" (catalog ID {gid}), labeled size {size}, intended cut: "{fit_style}", category: "{category}". Its flat-measurement specifications are: {garment_dims_text}. Reproduce every visible design element from the product reference — colors, logos, prints, buttons, seams, collar/waistband shape, cuff/hem treatment, and the fabric's weight and texture — without modification.

FIT ANALYSIS — For each body landmark, the delta (garment dimension minus person dimension) dictates how the fabric behaves on the body. Negative delta means the garment is smaller than the body, so the material must compress, stretch, or fail to cover. Positive delta means the garment is larger, so the material must hang loose, drape, or bunch. The render MUST physically reflect each of the following:

{fit_sections}

OVERALL SILHOUETTE — {silhouette}

RENDERING INSTRUCTIONS — Photograph-grade realism, not illustration. Simulate true textile physics: taut fibers and tension lines where the garment is undersized; clean relaxed lines where true-to-size; soft gravity-driven folds and bunching where oversized. Match the lighting direction, color temperature, and shadow softness of the person reference. Keep skin tones accurate, edges sharp, and every printed or embroidered detail on the garment legible. No cartoon, painterly, or stylized effects; no background change; no accessory additions. Output one final image only."""

    return prompt


def _overall_silhouette(deltas: dict, fit_style: str) -> str:
    if not deltas:
        return f"Standard {fit_style.lower()} silhouette."
    avg = sum(deltas.values()) / len(deltas)
    worst = min(deltas.values())
    biggest = max(deltas.values())
    if avg <= -20:
        return (
            f"Garment is dramatically undersized for this wearer. The fabric is at or near its stretch "
            f"limit at multiple landmarks and reads unmistakably as the wrong size — overriding the "
            f"garment's intended '{fit_style}' cut completely. Keep the render plausible: too-small, "
            f"not torn."
        )
    if avg <= -7.5:
        return (
            f"Heavily undersized. Multiple landmarks show clear compression and visible tension lines, "
            f"reading as an unintentionally body-hugging version of the '{fit_style}' cut."
        )
    if avg < -1.25:
        return (
            f"Snug overall — fabric sits close to the body at most landmarks, reading as a tight "
            f"version of the intended '{fit_style}' cut."
        )
    if avg <= 4:
        return (
            f"True-to-size — the '{fit_style}' cut reads cleanly with no strain and no excess "
            f"fabric at any landmark."
        )
    if avg <= 12.5:
        return (
            f"Relaxed — fabric sits slightly off the body everywhere, giving a gently oversized read "
            f"beyond the intended '{fit_style}' cut."
        )
    if avg <= 25:
        return (
            f"Oversized — fabric drapes clearly past the body in all directions with deep folds, "
            f"producing a streetwear-style dropped silhouette regardless of the intended '{fit_style}' cut."
        )
    return (
        f"Extremely oversized — the wearer is visibly drowning in fabric, with heavy pooling and "
        f"pleating throughout. Worst single delta {worst:+}, largest excess {biggest:+}."
    )


def _describe(dim: str, delta: float, person_val, garment_val, units: str) -> str:
    header = (
        f"• {dim.replace('_', ' ').upper()} "
        f"(person: {person_val} {units}, garment: {garment_val} {units}, "
        f"delta: {delta:+} {units}):"
    )

    if dim == "chest":
        if delta <= -38:
            body = (
                f"The garment is far too small at the chest — fabric is at its maximum possible stretch. "
                f"Render the cloth pulled taut with visible thinning of the weave, slight color washing where "
                f"it stretches across the body, ribcage and bust contour reading clearly through the material, "
                f"and any print or logo distorted and elongated. Keep the render plausible: a too-small "
                f"garment, not a torn or popped one."
            )
        elif delta <= -25:
            body = (
                f"The garment is dramatically undersized at the chest. Render heavy bodycon clinging with "
                f"pronounced horizontal tension lines from the underarm seams, side seams under obvious load "
                f"and bowing outward, prints visibly stretched, and any buttons or closures gaping under strain."
            )
        elif delta <= -15:
            body = (
                f"The chest is heavily strained. Pronounced tension lines radiate from the underarm seams, "
                f"the ribcage and bust outline read through the cloth, side seams visibly bow, and any chest "
                f"print or logo shows clear distortion from the stretch."
            )
        elif delta <= -7.5:
            body = (
                f"The chest is clearly tight. Faint horizontal tension lines appear from the underarm seams, "
                f"the body contour reads through the fabric, and the side seams begin to bow outward under load."
            )
        elif delta < -2.5:
            body = (
                f"The chest is snug. The fabric sits in light contact with the torso following the bust and "
                f"rib curve, without true stretch lines but with no air between body and cloth."
            )
        elif delta <= 2.5:
            body = (
                f"The chest is true to size. The fabric kisses the body and falls in a clean vertical line "
                f"from the shoulders to the ribcage, neither clinging nor puffing outward."
            )
        elif delta <= 7.5:
            body = (
                f"The chest is relaxed. A small air gap separates the fabric from the torso, with soft "
                f"straight drape lines from the shoulder. No tension anywhere."
            )
        elif delta <= 15:
            body = (
                f"The chest is loose. Soft vertical folds fall from the shoulder seams, with the fabric "
                f"clearly off the body and no body contour reading through."
            )
        elif delta <= 30:
            body = (
                f"The chest is heavily oversized. Render deep vertical folds from the shoulder seams, fabric "
                f"hanging well off the body in a pronounced tent-leaning silhouette."
            )
        else:
            body = (
                f"The chest is extremely oversized — the wearer is drowning in fabric. Render heavy pleating "
                f"and pooling all around the torso, with the garment clearly several sizes too large."
            )

    elif dim == "shoulder":
        if delta <= -15:
            body = (
                f"The shoulders are dramatically too narrow — the garment fits like it is one or two sizes too "
                f"small at the yoke. The seam approaches the collarbone, fabric is at maximum lateral stretch "
                f"across the upper back, and the armhole is high and severely constricted. Keep the render "
                f"plausible: too small, not torn."
            )
        elif delta <= -7.5:
            body = (
                f"The shoulders are severely tight. Seams sit well inside the shoulder bone, sleeves are "
                f"yanked upward, the armhole reads high and constricted, and visible strain pulls the fabric "
                f"toward the neckline."
            )
        elif delta < -2.5:
            body = (
                f"The shoulders are noticeably tight. Seams sit inboard of the shoulder bone, the sleeve heads "
                f"pull mildly upward, and faint strain lines read across the upper back."
            )
        elif delta <= 2.5:
            body = (
                f"The shoulders are aligned. Seams sit exactly on the shoulder bone, giving a clean tailored "
                f"line from neck to sleeve."
            )
        elif delta <= 7.5:
            body = (
                f"The shoulders sit slightly dropped. Seams fall just past the shoulder bone, giving a "
                f"relaxed casual line with mild fabric softening at the seam."
            )
        elif delta <= 15:
            body = (
                f"The shoulders are clearly dropped. Seams fall down the upper arm — several centimetres past the "
                f"natural shoulder edge — with pronounced fabric pooling at the armhole and a streetwear-leaning "
                f"oversized line."
            )
        else:
            body = (
                f"The shoulders are extremely dropped. Seams sit at or below the upper bicep, the armhole "
                f"hangs low and loose, and the silhouette reads as a tent regardless of the intended cut."
            )

    elif dim == "length":
        if delta <= -25:
            body = (
                f"The garment is extremely short for the wearer's torso. Render the hem high on the chest, "
                f"with the entire midriff and most of the lower ribcage exposed — the garment reads almost "
                f"like a bandeau or under-bust crop on this body. Keep it plausible, do not redraw it as a "
                f"different garment."
            )
        elif delta <= -15:
            body = (
                f"The garment is heavily cropped on this wearer. Render the hem at the upper ribcage with a "
                f"large section of midriff skin exposed between the hem and the waistband."
            )
        elif delta <= -7.5:
            body = (
                f"The garment is cropped on this wearer. The hem sits at the lower ribcage with clear midriff "
                f"exposure between the hem and the waistband."
            )
        elif delta < -2.5:
            body = (
                f"The garment is short. The hem sits just above the natural waist, revealing a narrow strip "
                f"of skin or visible waistband below the hem."
            )
        elif delta <= 2.5:
            body = (
                f"The length is correct. The hem sits cleanly at the natural waistline with a horizontal line "
                f"across the waist."
            )
        elif delta <= 7.5:
            body = (
                f"The garment runs long. The hem drops past the hipbone and fully covers the waistband, with "
                f"a small amount of fabric pooling near the hips."
            )
        elif delta <= 15:
            body = (
                f"The garment is tunic-length on this wearer. The hem falls to the upper thigh with visible "
                f"vertical folds along the torso."
            )
        else:
            body = (
                f"The garment is dress-length on this wearer. The hem falls toward the knee, producing an "
                f"elongated dress-like silhouette with heavy vertical drape."
            )

    elif dim == "arm_length":
        if delta <= -15:
            body = (
                f"The sleeves are dramatically short — they barely exist on this wearer. Render the sleeve "
                f"hems sitting near the shoulder cap, so the garment reads as effectively sleeveless or cap-"
                f"sleeved regardless of its original sleeve type. Keep the render plausible."
            )
        elif delta <= -7.5:
            body = (
                f"The sleeves are very short on this wearer. Render the sleeve hems barely past the shoulder, "
                f"with most of the upper arm or full forearm exposed beyond the cuff."
            )
        elif delta < -2.5:
            body = (
                f"The sleeves are clearly short. The hem ends visibly short of the intended landmark — "
                f"forearm or upper arm exposed beyond the cuff by a small but obvious margin."
            )
        elif delta <= 2.5:
            body = (
                f"The sleeves end at the correct landmark — exactly where the garment's cut intends them to "
                f"finish on the arm."
            )
        elif delta <= 7.5:
            body = (
                f"The sleeves run long. Fabric bunches at the cuff or extends past the wrist, with the hem "
                f"partially covering the back of the hand."
            )
        elif delta <= 15:
            body = (
                f"The sleeves are very long. The cuff fully covers the wrist and most of the back of the hand, "
                f"with visible fabric bunching above the cuff."
            )
        else:
            body = (
                f"The sleeves are extremely long. The cuff falls past the fingertips with heavy bunching all "
                f"along the forearm, producing an exaggerated oversized look."
            )

    elif dim in ("waist", "hip"):
        zone = "waistband" if dim == "waist" else "hip"
        if delta <= -15:
            body = (
                f"The {zone} is heavily strained. The {zone} digs into the body, side seams under load, "
                f"prints distorted, fabric visibly stretched. Keep render plausible — no tearing."
            )
        elif delta <= -7.5:
            body = (
                f"The {zone} is clearly tight. Visible {zone} impression, fabric strained around the seat "
                f"and hips, body contour reading clearly through the cloth."
            )
        elif delta < -2.5:
            body = (
                f"The {zone} is snug. The {zone} presses lightly into the body with a faint impression visible."
            )
        elif delta <= 2.5:
            body = f"The {zone} fits true to size — the garment sits cleanly without slipping or pinching."
        elif delta <= 7.5:
            body = f"The {zone} is relaxed — the garment sits comfortably with no compression."
        elif delta <= 15:
            body = (
                f"The {zone} is loose. The {zone} sags, a gap is visible at the back, the garment hangs "
                f"slightly off the hip."
            )
        else:
            body = (
                f"The {zone} is heavily oversized. The {zone} sits well off the body and the garment "
                f"clearly looks like the wrong size — render belt-cinched if appropriate."
            )

    elif dim in ("inseam", "outseam"):
        if delta <= -15:
            body = (
                f"The {dim} is extremely short — the hem rides near the knee or upper calf, garment looks "
                f"like a different cut entirely. Keep render plausible."
            )
        elif delta <= -7.5:
            body = f"The {dim} is very short — hem near mid-calf, garment reads as cropped."
        elif delta < -2.5:
            body = (
                f"The {dim} is short — hem rides above the ankle bone (ankle-pant look) regardless of "
                f"intended cut."
            )
        elif delta <= 2.5:
            body = f"The {dim} is correct — hem breaks cleanly at the ankle."
        elif delta <= 7.5:
            body = f"The {dim} runs long — fabric stacks at the ankle with one or two folds above the shoe."
        elif delta <= 15:
            body = f"The {dim} is very long — heavy fabric stacking covers the top of the shoe."
        else:
            body = f"The {dim} is extreme — hem drags or pools on the ground."

    elif dim == "thigh":
        if delta <= -15:
            body = (
                f"The thigh is heavily strained — prints distorted, side seams under load, fabric at "
                f"high stretch. Keep render plausible."
            )
        elif delta <= -7.5:
            body = (
                f"The thigh is clearly tight — clear thigh contour reads through the fabric with horizontal "
                f"tension lines on movement-prone areas."
            )
        elif delta < -2.5:
            body = (
                f"The thigh is snug — fabric in contact with the leg, light tension visible front and back."
            )
        elif delta <= 2.5:
            body = f"The thigh fits cleanly — fabric follows the leg without strain or excess."
        elif delta <= 7.5:
            body = f"The thigh is relaxed — slight air gap around the leg."
        elif delta <= 15:
            body = f"The thigh is loose — clear drape, baggy line from hip to knee."
        else:
            body = (
                f"The thigh is heavily oversized — deep folds and a wide-leg silhouette regardless of the "
                f"intended cut."
            )

    else:
        body = f"Delta at this landmark is {delta:+} {units}; reflect that magnitude in fabric behavior."

    return f"{header} {body}"


if __name__ == "__main__":
    sample = {
        "garment_id": "G2",
        "garment_name": "Adidas Navy Striped Bodysuit",
        "garment_size": "S",
        "garment_fit_style": "Bodycon",
        "category": "top",
        "units": "cm",
        "person_dimensions": {"length": 66, "chest": 91, "shoulder": 38, "arm_length": 23},
        "garment_dimensions": {"length": 71, "chest": 81, "shoulder": 36, "arm_length": 18},
        "deltas": {"length": 5, "chest": -10, "shoulder": -2, "arm_length": -5},
    }
    out = build_fit_prompt(sample)
    print(out)
    print(f"\n--- word count: {len(out.split())} words ---")
