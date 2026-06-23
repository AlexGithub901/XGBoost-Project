import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "Air_temperature_K": 298.1,
        "Process_temperature_K": 308.6,
        "Rotational_speed_rpm": 1500,
        "Torque_Nm": 40.0,
        "Tool_wear_min": 100
    }
)
print(response.json())