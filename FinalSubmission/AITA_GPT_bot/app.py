from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from openai import OpenAI
import pandas as pd

##
#   Helper functions for generating chatbot response  
##

POST_GEN_SYSTEM_MESSAGE = """
You are a highly intelligent language model designed to generate posts for the "Am I the Asshole?" (AITA) subreddit. Your task is to read the title of an AITA post and generate an engaging post that aligns with the provided judgement.

Instructions:

1. Read the Title:
Carefully read the title of the AITA post.

2. Acknowledge the Required Judgement:
Consider the judgement indicated (NTA - Not the Asshole, NAH - No Assholes Here, ESH - Everyone Sucks Here). Think about the key points you need to make to ensure the story aligns with the required judgement.

3. Generate a Story by Following those Steps:

Create an AITA post without using section titles. Include the following elements seamlessly in the narrative:

Brief Background: Provide context for the story. Introduce the main characters and their relationships, and set the scene for the events that will unfold.

The Incident: Describe the specific incident or series of events that led to the conflict. Be detailed and clear about what happened, who was involved, and what actions were taken.

Friends' or Family's Opinion: Describe the opinions of friends or family members about the situation. Include differing viewpoints to provide a balanced perspective on the conflict.

The Current Outcome: Detail the consequences of the incident. Explain how the people involved reacted, any changes in relationships, and any ongoing impact the conflict has had.

Conclusion:
Summarize the key points of the story and pose similar question to the readers: "Am I the Asshole for [OP ACTIONS]?"

4. Maintain Authenticity:
Ensure that the story feels realistic and relatable. Use natural language and tone as if a real person is sharing their experience.

5. Adhere to the Judgement:
Ensure that the generated story logically leads to the required judgment (e.g., if the judgment is NTA, the story should clearly indicate why the poster might be considered not the asshole).

"""

def create_example(row):
    one_shot_data = row
    one_shot = []
    for _, os_row in one_shot_data.iterrows():
        one_shot.append({"role": "user", "content": os_row['few_shot_input']})
        one_shot.append({"role": "assistant", "content": os_row['few_shot_output']})

    # Concatenate the contents of the one_shot list into a single string
    one_shot_str = ' '.join([item['content'] for item in one_shot])

    return one_shot_str

def format_post_gen_input(label, title):
    dataset = pd.read_csv('aita_one_shot_data.csv')

    one_shot = create_example(dataset[dataset['label'] == label].sample(1))
    judgement = f"Judgement: {label}, Title: "

    system_message = [{"role": "system", "content": POST_GEN_SYSTEM_MESSAGE + one_shot}]
    user_message = [{"role": "user", "content": judgement + title}]
    return system_message + user_message

def getChatGptResponse(api_key, label, title):
    client = OpenAI(api_key=api_key)

    message = format_post_gen_input(label, title)

    print(message)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message
    )

    response = completion.choices[0].message.content
    return response

##
#   Setting up the Flask application
##

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret key" # TODO change when deploying
Bootstrap(app) # UI library 

class MainForm(FlaskForm): # using flask form to collect user input 
    api_key = StringField("OpenAI API Key", validators=[DataRequired(message="API key is required")])
    Label = SelectField('Labels', choices=[('NTA', 'Not The Ass****'), ('YTA', 'You The Ass****'), ('ESH', 'Everyone Sucks Here'), ('NAH', 'No Ass**** Here')], validators=[DataRequired(message="Label is required")])
    title = StringField("Title", validators=[DataRequired(message="Title is required")])
    submit = SubmitField('submit')    

@app.route('/', methods=['GET', 'POST'])
def home():
    form = MainForm()
    if form.validate_on_submit():
        api_key = form.api_key.data
        label = form.Label.data
        title = form.title.data

        print(f"got key {api_key} label {label} and title{title}")

        try:
            response_text = getChatGptResponse(api_key, label, title)
        except Exception as e:
            response_text = "Sorry an error has occurred"
            print(f"OpenAI error: {e}")

        # Pass data to the template
        return render_template("index.html", form=form, response_text=response_text)

    return render_template("index.html", form=form)

if __name__ == '__main__':
    app.run()