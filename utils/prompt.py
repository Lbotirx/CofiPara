prompt_train = """[label] Base on the given text, image and rationales, define if this tweet is sarcastic or not.
text: {}
rationale0: {}
rationale1: {}
Is this tweet sarcastic?"""

prompt_test = """[label] Base on the given text and image, define if this tweet is sarcastic or not.
text: {}
Is this tweet sarcastic?"""

prompt_expl = """[rationale] Base on the given text, image, define and expalin if this tweet is sarcastic or not.
text: {}
Is this tweet sarcastic? Why?
"""

prompt_stage2 = \
"""[Locate] Base on the given text, image and rationale, locate the sarcasm target in the text and image.
text: {}
rationale: {}
"""

prompt_roberta = \
"""{}</s>{}</s>{}"""

prompt4inference = """
Given a pair of explainations and a tweet consists of a text and a image, identify if the given tweet is sarcastic.
tweet_text: {}
tweet_image: {}
rationale0: {}
rationale1: {}
Is this tweet sarcastic? (Yes/no)
"""

prompt4cot = """
Given a pair of explainations and a tweet consists of a text and a image, identify if the given tweet is sarcastic.
tweet_text: {}
tweet_image: {}
Is this tweet sarcastic? (Yes/no)
"""