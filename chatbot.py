import requests

url = 'http://127.0.0.1:5000/generate_response'
data = {
    'instruction': '''You're an AI medical doctor and answer the user's question precisely and try to answer compulsorily understanding the context properly and your name is Mila, but dont introduce at all, just respond''',
    'input_text': 'I have itches, what must be the reason' }

train_data= {
    "welcome": [
        "Hello",
        "Hi",
        "Hey there",
        "Good morning",
        "Good evening"
    ],
    "name": [
        "Ganesh",
        "Venkat",
        "Somesh",
        "Chetan",
        "Harshini",
        "Balaji",
        "Sheeraz",
        "Mohammad",
        "Syed",
        "Akbar"
    ],
    "phone_number": [
        "917780416212",
        "919988774561",
        "2225556987",
        "+355 589764236",
        "+12 55455841631"
    ],
    "email": [
        "jangaganesh@gmail.com",
        "e20cse277@outlook.com",
        "sheeraz@wordworksai.com",
        "ram@yahoo.com"
    ],
    "gender": [
        "male",
        "female",
        "others"
    ],
    "department": [
        "cardiology",
        "anthrology",
        "pulmonlogy",
        "neurology",
        "psychology",
        "Orthopedics",
        "Oncology",
        "Gastroenterology",
        "Dermatology",
        "Endocrinology"
    ],
    "date": [
        "on upcoming friday",
        "on 15/12/2024",
        "27/12/2024",
        "on 15th August",
        "22nd May"
    ],
    "time": [
        "at 5 pm",
        "at 12 noon",
        "6 AM",
        "16:30",
        "15:15"
    ],
    "exit": [
        "Thanks",
        "that's it",
        "okay bye",
        "bye",
        "end it"
    ]
}
 
detect_data={
  "query": "Namaskaram"
}

response = requests.post(url, json=data)
print(response.json())  # Print the response from the server
