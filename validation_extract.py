import logging
from pathlib import Path
from structured_data_extraction.main import process_single_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation_extract")

VALIDATION_DIR = Path(__file__).resolve().parent / "validation-set"

KEYS = {
    "0150200051": ["product family","cable length", "product specification", "Voltage", "net weight"],
    "2D iPSC p7 d0 222d119683938118b7f7d1f17fc21466": ["DMEM","mTeSR", "StemMACS"],
    "2D iPSC p7 d3 p8 d0 222d11968393813abd5de2b3a86d09c2": ["protocol","seeded"],
    "AD8-0000-E01": ["logic voltage", "supply current","humidity limits"],
    "AD840": ["gain bandwidth","quiescent current","input voltage noise"],
    "ada4523-1": ["internal chopping frequency", "EMF", "Output Voltage Swing Low"],
    "ALD1722_E": ["rise time", "overshoot factor", "Average Long term input offset"],
    "ALD2711": ["rise time", "PSRR", "slew rate"],
    "BC546": ["collector cutoff current","DC current gain","Thermal Resistance"],
    "Diff d-0 222d11968393818e8b47f1429be49193": ["CHIR","serological pipette"],
    "Diff d12 222d11968393810ea8b5d3a5bbfb24e1": ["lactate stock","serological pipette"],
    "Diff d1 222d1196839381baa62bea713941d2f4": ["albumin","falcon"],
    "Diff d-1 222d1196839381faaefdca27cb30255b": ["aliquot volume"],
    "Diff d14 222d119683938174ad21db3320bcaaa7": ["lactate selection medium", "comment"],
    "Diff d20 222d1196839381438012c3c03bc4af26": ["cardiac culture medium", "serological pipette"],
    "Diff d21 222d11968393810fa4a3ec2fecb80b09": ["PSC", "DNase", "Centrifuge","Cardiac Culture", "RPM","counts","BCT","purity"],
    "Diff d2 222d1196839381169b8dd844acd1f341": ["serological pipette"],
    "Diff d-2 222d11968393816ab739f96052bc35fa": ["ROCK inhibitor", "Incubated","detach", "RT PBS","mio per plate","aggregate cells"],
    "Diff d5 222d11968393812c9f44f4153ae70998": ["albumin","date"],
    "Diff d7 222d119683938193b326d3e17d4031f4": ["CM", "day 7"],
    "Diff d9 222d119683938107bbb6ea581e9f1820": ["freezer"],
    "eMT d1 2D CMs d0 222d11968393816eb11cc3fc5d43b60f": ["BCT", "DMEM","fibronectin"],
    "eMT d27 222d11968393810dba38caf880faf6df": ["16.1", "16.2"],
    "eMT d3 2D CMs d2 222d1196839381cf9bbbe4db5c387a95": ["eMT16.1","eMT16.2", "eMT day", "2D layer"],
    "eMT d63 222d11968393813db7bdede059f4319a": ["Date", "EMT medium"],
    "lf356": ["Input offset voltage", "Still Air", "Input Capacitance"],
    "LM321-D": ["Output Voltage","Phase Margin"],
    "lmv358a": ["CMRR", "Voltage Output Swing"],
    "rhf310a": ["Heavy Ions", "TID", "Maximum Power Dissipation"],
    "TL431": ["Min Cathode Current", "a1", "F"],
    "TLV27x": ["Input Bias Current","VOH","Power Supply Rejection Ratio"],
    "tsb951": ["Junction Temperature", "ESD", "SR", "GBP"],
}


def main():
    pending = {stem: keys for stem, keys in KEYS.items() if keys}
    if not pending:
        logger.error("No keys filled in — populate KEYS and re-run.")
        return

    logger.info(f"Running extraction for {len(pending)} file(s)")
    done, failed = [], []
    for stem, keys in pending.items():
        pdf_path = VALIDATION_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            logger.error(f"Missing PDF, skipping: {pdf_path.name}")
            failed.append(stem)
            continue
        try:
            result = process_single_pdf(str(pdf_path), keys)
            (done if result else failed).append(stem)
        except Exception as e:
            logger.error(f"Failed {stem}: {e}")
            failed.append(stem)

    logger.info(f"Done: {len(done)} succeeded, {len(failed)} failed/skipped")
    if failed:
        logger.info(f"Not extracted: {failed}")


if __name__ == "__main__":
    main()
