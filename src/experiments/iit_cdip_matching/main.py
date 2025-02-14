import re
import os
import argparse
import xml.etree.ElementTree as ET

def parse_xml_file(file_path, output_dir):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Remove namespaces in order to parse special tags
        content = re.sub(' xmlns="[^"]+"', '', content, count=1)
        content = re.sub(' xmlns:sd="[^"]+"', '', content, count=1)

        root = ET.fromstring(content)

        # Use specific tags to retrieve file paths
        for record in root.findall("record"):
            path_element = record.find("path")
            ocr_content = record.find("LTDLWOCR")

            if path_element is not None:
                path = path_element.text.strip()
                content = ""
                for elem in record:
                    # Check for the LTDLWOCR tag that contains the extracted text
                    if elem.tag != "path" and elem.tag != "LTDLWOCR":
                        content += elem.text.strip() if elem.text else ""
                if ocr_content is not None:
                    for child in ocr_content:
                        content += child.text.strip() if child.text else ""

                # Supress punctuation and remove whitespaces from text
                content = re.sub(r'[^\w\s]', '', content).strip()
                # Compute output path that is the RVL-CDIP corresponding image path
                new_txt_path = os.path.join(output_dir, path).split("/")[:-1]
                try:
                    # Create a new txt file for each image of the RVL-CDIP dataset
                    new_txt_path = os.path.join(*new_txt_path)
                    with open(new_txt_path + "/ocr.txt", 'w', encoding='utf-8') as f:
                        f.write(content)
                except:
                    continue
    except ET.ParseError as pe:
        print(f"Error parsing XML file: {file_path}")
        print(f"ParseError message: {str(pe)}")
    except ValueError as ve:
        print(f"Error processing XML file: {file_path}")
        print(f"ValueError message: {str(ve)}")
    except Exception as e:
        print(f"Error processing XML file: {file_path}")
        print(f"Error message: {str(e)}")

def process_ocr_files(folder_path, output_dir):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Loop through each file and folder of the root folder. It first checks if it's a file or a folder
            if os.path.isfile(file_path):
                # Check if the current file is a xml file
                if re.search(r"\.xml$", file):
                    try:
                        print(f"Parsing {file_path}")
                        parse_xml_file(file_path, output_dir)
                    except Exception as e:
                        print(f"Error processing XML file: {file_path}")
                        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    args = parser.parse_args()

    # Call the main function indicating input folder (IIT-CDIP) and output folder (RVL-CDIP)
    process_ocr_files(args.input_dir, args.output_dir)
