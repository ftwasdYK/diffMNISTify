import numpy as np
import requests
import os
import json
import argparse
from PIL import Image

PORT = 8000
ENDPOINT = f"http://localhost:8000/generate"
TIMEOUT_SECONDS = 600

def main(args):
# Send requests to the API #
    number_req  = args.num   
    REQUEST_BODY = {
        "number": number_req
    }

    response = requests.post(ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )

    if response.ok:
        # with open(output_filename, "w") as outfile:
        data = response.json()
        img_array = np.array(data["img_list"])
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save("./img_gen_response.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True, help="The Number which will be generated from the request")
    args = parser.parse_args()
    main(args)