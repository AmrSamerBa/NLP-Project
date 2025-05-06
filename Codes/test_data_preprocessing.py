import re
import pandas as pd

def clean_reddit_post(text):
    text = re.sub(r'\r\n|\r', '\n', text).strip()

    main_body = re.split(r'\n*edit[:\-]?', text, flags=re.IGNORECASE)[0].strip()

    tldr_match = re.search(r'(?:tl;?dr[:\-]?\s*)(\S.*)', main_body, re.IGNORECASE | re.DOTALL)
    summary = tldr_match.group(1).strip() if tldr_match else ""

    main_body = re.sub(r'(?:tl;?dr[:\-]?\s*\S.*)', '', main_body, flags=re.IGNORECASE | re.DOTALL)

    return main_body.strip(), summary


with open("tldr_test_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

posts = re.split(r'TEST DATA #[0-9]+', raw_text)
posts = [p.strip() for p in posts if p.strip()] 

cleaned = [clean_reddit_post(p) for p in posts]
df = pd.DataFrame(cleaned, columns=["clean_content", "summary"])

df.to_csv("Test_data.csv", index=False)
