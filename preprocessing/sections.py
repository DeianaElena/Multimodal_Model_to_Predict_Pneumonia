#list of recurrent words/sentences that represent main categories of discharge notes

accepted_list = [
    "Sex",           #adding this to make sur it includes this info
    "MEDICINE",
    "Adverse Drug Reactions",
    "Chief Complaint",
    "Past Medical History",
    "Social History",
    "Physical Exam",
    "Pertinent Results",
    "ADMISSION PHYSICAL EXAM",
    "ADMISSION LABS",
    "CHRONIC ISSUES",
    "ADMISSION",
    "ADMISSION EXAM",
    "Admission Labs",
    "Transitional Issues"
]


rejected_list = [
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
    "FINAL REPORT ___",
    "IMAGING",
    "DISCHARGE PHYSICAL EXAM",
    "DISCHARGE LABS",
    "TRANSITIONAL ISSUES",
    "MEDICATIONS",
    "STUDIES",
    "CXR ___",
    "PRIMARY DIAGNOSIS",
    "DISCHARGE EXAM",
    "DISCHARGE",
    "Imaging",
    "Discharge Physical Exam",
    "Extended Care",
    "IMPRESSION",
    "Primary Diagnosis",
    "Discharge Labs",
    "ACUTE/ACTIVE ISSUES:",
    "ACTIVE ISSUES",
    "ACUTE ISSUES",
    "Brief Hospital Course",
    "MICROBIOLOGY",
    "CXR",
    "U/S"

]