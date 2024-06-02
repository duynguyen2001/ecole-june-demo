# %%
import requests

# Define the API endpoint
url = "http://localhost:16008/is-concept-in-image"

# Define the data to be sent in the POST request
data = {"user_id": "example_user_id", "threshold": "0.7", "concept_name": "biplane"}

# Specify the image file to be uploaded
image_path = (
    "/shared/nas2/knguye71/ecole-june-demo/samples/DemoJune2024-2/biplane/000002.jpg"
)
files = {"image": open(image_path, "rb")}

# Send the POST request
response = requests.post(url, data=data, files=files)

# Check the response
if response.status_code == 200:
    print("Request was successful")
else:
    print("Request failed")
    print("Status code:", response.status_code)
    print("Response text:", response.text)

# %%

# Check if the request was successful
if response.status_code == 200:
    # Stream the response content to a JSON file
    with open("response.json", "w") as json_file:
        json_file.write(response.text)
    print("Response content saved to response.json")
else:
    print("Request failed")
    print("Status code:", response.status_code)
    print("Response text:", response.text)

# %%
