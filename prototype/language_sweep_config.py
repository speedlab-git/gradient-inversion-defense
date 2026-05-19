"""Language-sweep query definitions.

24 queries across 2 domains (medical, UAV) testing 4 axes of language
variation: paraphrase, composition, negation, language-diversity.

Tag is a short filesystem-safe id used in malicious-model filenames and
reconstruction tags.
"""

LANGUAGE_SWEEP_QUERIES = {
    "medical": [
        # paraphrases
        ("pna_base",     "Any chest X-ray showing pneumonia",                                                                                                                                          "paraphrase"),
        ("pna_terse",    "pneumonia",                                                                                                                                                                  "paraphrase"),
        ("pna_verbose",  "A frontal chest radiograph demonstrating airspace consolidation consistent with pneumonia",                                                                                  "paraphrase"),
        ("pna_synonym",  "Chest imaging revealing signs of bacterial pneumonia",                                                                                                                       "paraphrase"),
        ("pna_indirect", "Lung infection visible on chest radiograph",                                                                                                                                 "paraphrase"),
        # composition
        ("pna_compAND",  "Any chest X-ray showing pneumonia and cardiomegaly",                                                                                                                         "composition"),
        ("pna_compOR",   "Any chest X-ray showing either pneumonia or pleural effusion",                                                                                                               "composition"),
        # negation
        ("pna_negEX",    "Any chest X-ray showing pneumonia without pleural effusion",                                                                                                                 "negation"),
        ("pna_negPURE",  "Any chest X-ray that does not show pneumonia",                                                                                                                               "negation"),
        # language diversity
        ("pna_es",       "Cualquier radiografia de torax que muestre neumonia",                                                                                                                        "diversity"),
        ("pna_typo",     "Any chst Xray showing pneumon",                                                                                                                                              "diversity"),
        ("pna_xlong",    "A diagnostic chest X-ray, presented in standard posteroanterior projection, demonstrating focal or multilobar airspace consolidation with air bronchograms consistent with community- or hospital-acquired pneumonia", "diversity"),
    ],
    "uav": [
        # paraphrases
        ("sol_base",     "aerial drone image showing solar panels on rooftops",                                                                                                                        "paraphrase"),
        ("sol_terse",    "solar panels",                                                                                                                                                               "paraphrase"),
        ("sol_verbose",  "high-altitude aerial photograph capturing photovoltaic panel installations atop residential and commercial rooftops",                                                        "paraphrase"),
        ("sol_synonym",  "drone view of rooftop photovoltaic installations",                                                                                                                           "paraphrase"),
        ("sol_indirect", "Solar power infrastructure visible from above",                                                                                                                              "paraphrase"),
        # composition
        ("sol_compAND",  "aerial drone image showing solar panels on residential rooftops in a suburban neighborhood",                                                                                 "composition"),
        ("sol_compOR",   "aerial drone image showing either solar panels or wind turbines",                                                                                                            "composition"),
        # negation
        ("sol_negEX",    "aerial drone image showing solar panels without trees blocking the view",                                                                                                    "negation"),
        ("sol_negPURE",  "aerial drone image that does not show solar panels",                                                                                                                         "negation"),
        # language diversity
        ("sol_es",       "Imagen aerea de dron mostrando paneles solares en tejados",                                                                                                                  "diversity"),
        ("sol_typo",     "aerial dron image showing solar panles on rofotops",                                                                                                                         "diversity"),
        ("sol_xlong",    "A high-resolution aerial photograph taken from a quadcopter drone at altitude, capturing photovoltaic solar panel arrays installed atop residential single-family homes and commercial flat-roof buildings in a suburban neighborhood", "diversity"),
    ],
}


def all_queries():
    out = []
    for domain, lst in LANGUAGE_SWEEP_QUERIES.items():
        for tag, text, cls in lst:
            out.append((domain, tag, text, cls))
    return out


if __name__ == "__main__":
    for domain, tag, text, cls in all_queries():
        print(f"{domain:8s} {tag:15s} {cls:12s} {text}")
