from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from better_profanity import profanity
import torch
import re

DetectorFactory.seed = 0

lang_dect = {
    "ar": "arb_Arab",
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "zh-cn": "zho_Hans",
    "fr": "fra_Latn"
}

lang_tgt = {
    "arabic": "arb_Arab",
    "english": "eng_Latn",
    "russian": "rus_Cyrl",
    "mandarin": "zho_Hans",
    "french": "fra_Latn"
}

def detect_language(text):
    try:
        language = detect(text)
        return lang_dect.get(language, "eng_Latn")
    except LangDetectException:
        return "Could not detect language"

def translate_nllb(text, src_lang_code, tgt_lang_code, tokenizer, model, device="cuda"):

    model = model.to(device)
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nllb_model = nllb_model.to(device)

post = """
TIFU play fighting with my bf, vomiting, then having an asthma attack
Me and my bf were lounging around the house yesterday. We call Sundays "slack Sundays" because we both have demanding jobs and Sunday is the only day we really get to chill out and do whatever we want. So we were wrapped up in a sheet on the couch browsing Netflix, no plans, no obligations, snacking on a very healthy and nutritious bowl of candy.

 However, a few hours into our do nothing session, he decided he wanted to do something. He said we should go hiking since the weather is warm, dry, and there was a nice breeze outside. Tbf, we don't get many opportunities to go hiking because of the unpredictable weather here... but the feeling of the cool comfortable sheet burrito. In my comfortable pajamas. Half asleep on the comfortable couch. I was in maximum relaxation mode. In other words, I wasn't about to get up any time soon, and told him. When he tried to persuade me with kisses, I said he would have to fight me out of bed (well, couch.)

This is how it happened. Instead of negotiating like a normal couple when we disagree on things, we like to wrestle, which is what we did. He jumped up and tried to drag me off the couch, while I tried to reel him back in. When he started gaining the upper hand, I decided to make a surprise attack as a last resort and knock him off his feet. So I wrapped my arms around his hips and pulled him down with me. Hard.

.......... I often underestimate our size difference. We're both big, muscular guys; He just has more fat and is 4 inches taller than me. I thought it would be fine as long as I braced myself, and shifted his weight toward the other side of the couch instead of my body.

I was given a very rude reality check.

I was strong enough to tug him down, but definitely not enough to shift his weight. Before he collapsed, he flipped over and ended up landing ass-first on my gut. A couple things happened very quickly: I felt all the wind being knocked out of my lungs and couldn't catch my breath. Then, I started feeling extremely nauseous. All the candy I ate, combined with the jostling from play fighting, and now his butt was grinding into my gut. Awful combination. Before I could shove him off of me, I felt the bile coming up and grabbed the closest vomit receptacle, which was the candy bowl that unfortunately was still half full. And just in time. I've never vomited so much in my life.

Bf immediately wiggled off me and stared at me, understandably shocked. Around the third round of puking, he ran to the kitchen to grab some napkins and I was mostly dry heaving at that point... Which made it difficult to breathe. Then I started wheezing, my airway felt damp, and I could feel my asthma flaring up. I stopped puking just long enough to croak out that I need my inhaler. He thankfully heard me and ran around to the usual places looking for my inhaler, but yelled that he couldn't find it, so he brought my nebulizer and Montelukast pouches instead.

When I finally stopped emptying my stomach, he wiped my mouth and snapped the mask over my face, and I got some some sweet, sweet oxygen back in my lungs. Don't remember much after that because the meds made me feel woozy.

Needless to say; We didn't go hiking, and we're going to be much more careful wrestling from now on.
"""

src_lang = detect_language(post)

target_lang_input = input("Enter a language to have the summary in (english, arabic, russian, mandarin, french):\n").lower()
tgt_lang = lang_tgt.get(target_lang_input, "eng_Latn")

if src_lang != 'eng_Latn':

    post = translate_nllb(
    	text=post,
    	src_lang_code=src_lang,
    	tgt_lang_code=tgt_lang,
    	tokenizer=nllb_tokenizer,
    	model=nllb_model,
    	device=device
    )

bart_model_path = "./bart-large-final"
summ_tokenizer = AutoTokenizer.from_pretrained(bart_model_path)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_path).to(device)

inputs = summ_tokenizer(post, return_tensors="pt", max_length=512, truncation=True).to(device)
summary_ids = summ_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

profanity.load_censor_words()

def custom_censor(text, level="full"):
    censored_text = text
    bad_words = profanity.CENSOR_WORDSET

    for word in bad_words:
        word_str = str(word)
        pattern = re.compile(rf"\b{re.escape(word_str)}\b", re.IGNORECASE)
        matches = pattern.findall(text)

        for match in matches:
            if level.lower() == "none":
                replacement = match
            elif level.lower() == "partial":
                if len(match) <= 2:
                    replacement = "*" * len(match)
                else:
                    replacement = match[0] + "*" * (len(match) - 2) + match[-1]
            elif level.lower() == "full":
                replacement = "*" * len(match)
            else:
                raise ValueError("Invalid censorship level. Choose from 'none', 'partial', or 'full'.")

            censored_text = re.sub(rf"\b{re.escape(match)}\b", replacement, censored_text, flags=re.IGNORECASE)

    return censored_text

censor_level = input("Enter your level of censorship (full, partial, none):\n")
summary = custom_censor(summary, censor_level)

if tgt_lang != 'eng_Latn':

    summary = translate_nllb(
    	text=summary,
    	src_lang_code=src_lang,
    	tgt_lang_code=tgt_lang,
    	tokenizer=nllb_tokenizer,
    	model=nllb_model,
    	device=device
	)

print("\nSummary:", summary)