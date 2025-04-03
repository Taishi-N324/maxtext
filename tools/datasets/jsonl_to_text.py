import json


def extract_text_from_jsonl(input_file, output_file):
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):

        for line in f_in:
            data = json.loads(line)
            new_data = {
                "text": data["text"],
            }
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write("\n")


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_text_from_jsonl(input_file, output_file)
