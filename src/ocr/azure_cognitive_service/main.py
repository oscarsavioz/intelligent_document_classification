import requests

# Replace 'key' with your subscription key.
key = 'b00ed9f4636b45328afd6555bdec19b5'

# Replace 'endpoint' with your endpoint URL.
endpoint = 'https://imgur.com/a/FclcLtz'

# The URL of the image you want to analyze.
image_url = 'https://www.dropbox.com/s/2h09gi05n9z1qhw/2505287241b.tif?dl=0'

headers = {'Ocp-Apim-Subscription-Key': key}
params = {'visualFeatures': 'Categories,Description,Color'}
data = {'url': image_url}
response = requests.post(endpoint, headers=headers, params=params, json=data)
response.raise_for_status()
# The 'analysis' object contains various fields that describe the image.
# The most relevant caption for the image is obtained from the 'description' property.
analysis = response.json()
print(analysis)
