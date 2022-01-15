from src.model import Model
import streamlit as st
import numpy as np
import os


PAGE_CONFIG = {"page_title":"App by Glad Nayak","page_icon":":white_check_mark:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

def main():
    """
    Render UI on web app
    """

    # load model
    model = Model()
    print('loaded all models and tokenizers')
    
    # adding the text that will show in the text box as default
    default_value = """
    LONDON, Jan 7 (Reuters) - Bitcoin slumped as much as 5% on Friday to its lowest since late September, amid a broader sell-off for cryptocurrencies driven by concerns about tighter U.S. monetary policy. Bitcoin was last down more than 3% at $41,704 after touching $40,938, its lowest since Sept. 29, as a mixed bag of U.S. payrolls data fuelled some bargain buying. The world's biggest cryptocurrency has lost over 40% since hitting a record high of $69,000 in November and the volatility that has plagued it since its birth 13 years ago remains stubbornly present. The global computing power of the bitcoin network has dropped sharply this week following the shutdown of Kazakhstan's internet as an uprising hit the country's fast-growing cryptocurrency mining industry. Bitcoin has also been under pressure after minutes from the latest U.S. Federal Reserve meeting, released on Wednesday, appeared to lean toward more aggressive policy action, sapping investor appetite for riskier assets. "We are seeing broad risk-off sentiment across all markets currently as inflationary concerns and rate hikes appear to be at the forefront of speculators' minds," said Matthew Dibb, COO of Singapore crypto platform Stack Funds. "Liquidity in BTC has been quite thin on both sides and there is risk of a retreat back to the mid-30's on the short term." Ether , the second largest token by market cap, fell as much as 8.6% to $3,114, its lowest since Oct. 1. It was last trading down more than 6% at $3,200.
    """

    st.title("Abstractive Summary and Keyword Extraction of Financial News")
    text = st.text_area("Paste an article to get started 🤗", default_value.strip(), height = 275)

    if text.strip():
        label, score = model.get_sentiment(text)
        if label == 'POSITIVE':
            st.write(f':simple_smile: This article has a {label.lower()} sentiment. Confidence score: {score:.2f}')

        else:
            st.write(f':cry: This article has a {label.lower()} sentiment. Confidence score: {score:.2f}')

        keywords, scores = model.get_keywords(text, with_highlight=False)
        st.subheader(f'Suggested Keywords:')
        st.write(', '.join(keywords))

        title = model.get_title(text)
        st.subheader(f'Suggested Headline:')
        st.write(title)

        summary = model.get_summary(text)
        st.subheader("And here's the abstract:")
        st.write(summary)

if __name__ == '__main__':
    main()