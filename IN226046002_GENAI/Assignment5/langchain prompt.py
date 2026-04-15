from langchain_core.prompts import (
    PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)

t1_pt = PromptTemplate(input_variables=["topic"], template="Explain {topic} in simple terms for beginners")
print(t1_pt.format(topic="Generative AI"))

t2_pt = PromptTemplate(input_variables=["topic", "audience", "tone"], template="Explain {topic} for {audience} in a {tone} tone")
tc_arr = [
    {"topic": "AI", "audience": "beginners", "tone": "friendly"},
    {"topic": "Python", "audience": "kids", "tone": "fun"},
    {"topic": "Deep Learning", "audience": "engineers", "tone": "technical"}
]
for tc in tc_arr:
    print(t2_pt.format(**tc))

t3_teach = PromptTemplate(input_variables=["topic"], template="Explain {topic} clearly step by step")
t3_int = PromptTemplate(input_variables=["topic"], template="Ask 3 questions about {topic}")
t3_story = PromptTemplate(input_variables=["topic"], template="Explain {topic} as a story")

top = "Machine Learning"
print(t3_teach.format(topic=top))
print(t3_int.format(topic=top))
print(t3_story.format(topic=top))

sys_msg = SystemMessagePromptTemplate.from_template("You are a {role}.")
usr_msg = HumanMessagePromptTemplate.from_template("Let's talk about {topic}.")
chat_pt = ChatPromptTemplate.from_messages([sys_msg, usr_msg])

res = chat_pt.format_messages(role="teacher", topic="Neural Networks")
for msg in res:
    print(f"{msg.type}: {msg.content}")

def val_inps(aud, tone):
    f_aud = aud if aud in ["beginner", "intermediate", "expert"] else "beginner"
    f_tone = tone if tone in ["formal", "casual", "fun"] else "casual"
    return f_aud, f_tone

def gen_prompt(top, aud, tone, style):
    v_aud, v_tone = val_inps(aud, tone)
    s_dict = {
        "teaching": "Explain {topic} for {aud} in a {tone} tone clearly step by step.",
        "interview": "Ask 3 questions about {topic} for {aud} in a {tone} tone.",
        "storytelling": "Explain {topic} for {aud} in a {tone} storytelling style."
    }
    dyn_pt = PromptTemplate(input_variables=["topic", "aud", "tone"], template=s_dict.get(style, s_dict["teaching"]))
    return dyn_pt.format(topic=top, aud=v_aud, tone=v_tone)

print(gen_prompt("Neural Networks", "beginner", "fun", "storytelling"))

t7_pt = PromptTemplate(input_variables=["concept", "app"], template="How does {concept} impact {app}?")
inps = [
    {"concept": "Computer Vision", "app": "self-driving cars"},
    {"concept": "NLP", "app": "customer service bots"},
    {"concept": "Reinforcement Learning", "app": "robotics"},
    {"concept": "GANs", "app": "digital art creation"},
    {"concept": "Predictive Analytics", "app": "healthcare diagnostics"}
]
for i in inps:
    print(t7_pt.format(**i))