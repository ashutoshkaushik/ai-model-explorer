"""'Surprise me': one AI fact at a time, posed as a question you reveal.

The 25 facts are written by hand (never generated at run time) so each one can be checked once and stays true.
"""

import random

import streamlit as st

from ui import chrome

# (year, topic, the curious question, the answer, why it matters)
FACTS = [
    (1950, "Origins", "Which paper asked “Can machines think?”, and then decided that question was too vague?",
     "Alan Turing's 1950 “Computing Machinery and Intelligence”. He replaced it with the “imitation game”, now "
     "called the Turing test, and predicted that by about 2000 a machine would fool an average interrogator 30% of "
     "the time after five minutes.",
     "75 years later, people still argue about whether passing it would prove anything."),
    (1950, "Origins", "What was the first “model” in this dataset, and did it have legs?",
     "Almost: Theseus (1950) was Claude Shannon's mechanical mouse at Bell Labs. Relays under a maze let it find "
     "the cheese by trial and error, then remember the route.",
     "Learning from experience and remembering it is still the core idea of machine learning."),
    (1951, "Hardware", "How many vacuum tubes did one of the first neural network machines need?",
     "Marvin Minsky's SNARC (1951) simulated about 40 neurons using roughly 3,000 vacuum tubes and parts from a B-24 "
     "bomber's autopilot.",
     "Today a single GPU runs networks with billions of parameters; the idea is the same, the scale is not."),
    (1955, "Origins", "When was the phrase “artificial intelligence” coined, and why?",
     "In a 1955 funding proposal by John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon, for a "
     "two-month summer workshop at Dartmouth in 1956.",
     "The proposal expected “a significant advance” in one summer. The field has been optimistic about timelines "
     "ever since."),
    (1958, "Hype", "What did a 1958 newspaper report the US Navy expected a computer to eventually do?",
     "Walk, talk, see, write, reproduce itself and be conscious of its existence. The machine was Frank Rosenblatt's "
     "Perceptron, a single layer of learnable weights.",
     "AI hype is older than most programming languages."),
    (1959, "Words", "Who coined the term “machine learning”, and with which board game?",
     "Arthur Samuel at IBM, in a 1959 paper on a checkers program that improved by playing against itself.",
     "Self-play came back, at vastly larger scale, in AlphaGo Zero almost 60 years later."),
    (1966, "People", "Why did a secretary ask her boss to leave the room so she could talk to a computer?",
     "She was chatting with ELIZA, Joseph Weizenbaum's 1966 program that mimicked a therapist by reflecting your own "
     "words back. Weizenbaum was alarmed at how readily people confided in it.",
     "People attribute understanding to fluent text. That's now called the ELIZA effect."),
    (1969, "Setbacks", "Which tiny logic puzzle helped stall neural network research for over a decade?",
     "XOR. Minsky and Papert's book “Perceptrons” (1969) showed a single-layer perceptron can't learn “one or the "
     "other but not both”. Multi-layer networks can, but nobody could train them well yet.",
     "The fix, backpropagation through hidden layers, only became popular in 1986."),
    (1973, "Setbacks", "What report triggered the first “AI winter” in Britain?",
     "The 1973 Lighthill Report, which judged that AI had failed to deliver on its promises. UK funding for most AI "
     "research was cut soon after.",
     "Cycles of hype and disappointment are part of AI's history, which is why this app shows eras."),
    (1986, "Methods", "Which 1986 paper taught neural networks to learn from their mistakes, layer by layer?",
     "Rumelhart, Hinton and Williams' Nature paper “Learning representations by back-propagating errors”. The idea "
     "had earlier roots, but this paper made it the standard way to train networks.",
     "Every large model in this dataset is still trained with backpropagation."),
    (1997, "Games", "How many chess positions per second did Deep Blue examine when it beat Kasparov?",
     "About 200 million. IBM's Deep Blue beat world champion Garry Kasparov 3½–2½ in a six-game match in 1997.",
     "It was brute-force search plus hand-tuned rules, not learning, which is why it didn't lead to general AI."),
    (1998, "Applications", "Where was a neural network quietly at work in the 1990s, long before the hype?",
     "Reading handwritten digits on bank cheques. Yann LeCun's convolutional networks (the LeNet family) were used "
     "commercially to process a sizeable share of US cheques by the late 1990s.",
     "Useful AI often arrives quietly, in narrow tasks, years before it makes headlines."),
    (2009, "Data", "Who labelled the 14 million images that kicked off the deep learning boom?",
     "Thousands of online crowd workers on Amazon Mechanical Turk, for Fei-Fei Li's ImageNet project.",
     "Data, not just algorithms, made deep learning work. Someone always pays for the labels."),
    (2012, "Breakthroughs", "How many GPUs did it take to start the deep learning revolution?",
     "Two. AlexNet was trained on two NVIDIA GTX 580 gaming cards for about a week, and cut ImageNet's top-5 error "
     "to 15.3% when the runner-up scored 26.2%.",
     "This app's growth fit starts in 2010 because compute trends changed sharply around then."),
    (2013, "Language", "What do you get if you compute “king − man + woman”?",
     "Something very close to “queen”. Google's word2vec (2013) learned word vectors where directions carry meaning, "
     "so arithmetic on words roughly works.",
     "Embeddings like these are the first layer of every language model today."),
    (2014, "Methods", "Which famous AI idea was reportedly sketched after an argument in a bar?",
     "Generative adversarial networks. Ian Goodfellow has said he came up with GANs after a discussion with friends "
     "at a Montreal bar in 2014, then coded the first version that night.",
     "Two networks competing, one generating and one judging, powered image generation for years."),
    (2016, "Games", "What were the odds a human would have played AlphaGo's famous “move 37”?",
     "About 1 in 10,000, by AlphaGo's own estimate. Commentators first thought the move in game 2 against Lee Sedol "
     "was a mistake; it helped win the game.",
     "Systems trained at scale can find strategies people had overlooked for centuries."),
    (2017, "Breakthroughs", "How long did it take to train the original Transformer?",
     "The base model in “Attention Is All You Need” trained for 12 hours on 8 GPUs; the big one took 3.5 days. "
     "The architecture now underpins nearly every large language model.",
     "One of the most influential AI papers ever trained on less hardware than many university labs have."),
    (2019, "Openness", "Which language model was first held back because it might be “too dangerous”?",
     "GPT-2 (1.5 billion parameters). OpenAI released it in stages through 2019, citing misuse concerns, before "
     "publishing the full model in November.",
     "The debate about open versus closed weights, charted in this app's Explorer, started in earnest here."),
    (2019, "Ideas", "What is “the bitter lesson” of 70 years of AI research?",
     "Rich Sutton's 2019 essay: general methods that scale with computation (search and learning) have beaten "
     "methods built on human knowledge, again and again.",
     "It's the argument behind the steep compute curve on this app's Explorer page."),
    (2020, "Scale", "How much bigger was GPT-3 than GPT-2?",
     "Over 100 times: 175 billion parameters versus 1.5 billion, a year apart. Training took roughly 3 × 10²³ FLOP.",
     "Find GPT-3 on the Explorer's scatter: it sits right on the trend line, not above it."),
    (2022, "Methods", "Can a smaller model beat one four times its size?",
     "Yes. DeepMind's Chinchilla (70B parameters) beat Gopher (280B) by training on more data: 1.4 trillion tokens. "
     "The rule of thumb became about 20 training tokens per parameter.",
     "Parameter count alone is a poor measure of capability, which is why this app plots compute too."),
    (2022, "Adoption", "How fast did ChatGPT reach 100 million users?",
     "About two months after its November 2022 launch, according to a widely cited UBS estimate. That made it "
     "one of the fastest-growing consumer apps ever at the time.",
     "Model progress became visible to everyone at once, not just researchers."),
    (2024, "Recognition", "Which Nobel Prizes went to AI in the same week?",
     "Physics: John Hopfield and Geoffrey Hinton, for foundations of neural networks. Chemistry: Demis Hassabis "
     "and John Jumper for AlphaFold's protein structure prediction, shared with David Baker for protein design.",
     "AI went from a field that needed a winter's patience to two Nobel Prizes in October 2024."),
    (2026, "This dataset", "How fast has training compute grown in this app's own data?",
     "About 4.3× per year since 2010, so it doubles roughly every six months. Moore's law doubled transistor "
     "counts about every two years.",
     "Open the Explorer Lab's growth-fit page to move the start year and watch the rate change."),
]


def surprise_page() -> None:
    n = len(FACTS)
    st.session_state.setdefault("fact_i", random.randrange(n))
    st.session_state.setdefault("fact_seen", set())
    i = st.session_state.fact_i
    st.session_state.fact_seen.add(i)
    year, topic, question, answer, why = FACTS[i]

    st.markdown("<div class='eyebrow'>Surprise me</div>", unsafe_allow_html=True)
    st.title("A question about AI you probably can't answer")
    st.markdown("<div class='lede'>Guess first, then reveal. 25 hand-picked facts from 75 years of AI.</div>",
                unsafe_allow_html=True)

    revealed = st.session_state.get("fact_revealed") == i
    body = (f"<div class='a'>{answer}</div><div class='why'><b>Why it matters · </b>{why}</div>" if revealed else "")
    st.markdown(f"<div class='fact'><div class='meta'><span>{topic} · {year}</span><span>Fact {i + 1} of {n}</span>"
                f"</div><div class='q'>{question}</div>{body}</div>", unsafe_allow_html=True)

    def go(to: int) -> None:
        st.session_state.fact_i = to % n
        st.session_state.fact_revealed = None

    def shuffle() -> None:
        unseen = [j for j in range(n) if j not in st.session_state.fact_seen] or [j for j in range(n) if j != i]
        go(random.choice(unseen))

    a, b, c, d = st.columns(4)
    a.button("← Previous", on_click=go, args=(i - 1,), width="stretch")
    if revealed:
        b.button("Hide answer", on_click=lambda: st.session_state.update(fact_revealed=None), width="stretch",
                 icon=":material/visibility_off:")
    else:
        b.button("Reveal the answer", type="primary", width="stretch", icon=":material/visibility:",
                 on_click=lambda: st.session_state.update(fact_revealed=i))
    c.button("Surprise me again", on_click=shuffle, width="stretch", icon=":material/shuffle:")
    d.button("Next →", on_click=go, args=(i + 1,), width="stretch")

    seen = st.session_state.fact_seen
    dots = "".join(f"<i class='{'on' if j == i else 'seen' if j in seen else ''}'></i>" for j in range(n))
    st.markdown(f"<div class='dots' title='Facts you have seen'>{dots}</div>", unsafe_allow_html=True)
    st.caption(f"You've seen {len(seen)} of {n}. “Surprise me again” picks one you haven't seen yet.")
    chrome.credit()
