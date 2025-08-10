def get_medication_info(med_name):
    meds = {
        "Paracetamol": "Used to treat pain and fever. Typical dose: 500mg twice daily.",
        "Ibuprofen": "Anti-inflammatory. Dose: 200–400mg every 6 hours."
    }
    return meds.get(med_name, "Medication info not found.")

def schedule_appointment(patient_id, doctor, date):
    return f"Appointment scheduled for patient {patient_id} with {doctor} on {date}."

def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return f"Your BMI is {bmi:.2f}."
