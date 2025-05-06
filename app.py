from  flask import Flask,request,render_template
import json
from flask_cors import CORS 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
conversation_history = []



@app.route('/', methods =['GET'])
def home():
    return render_template('index.html')


@app.route('/chatbot', methods = ['POST'])
def handle_prompt():
    data = request.get_data(as_text = True)
    data = json.loads(data)
    print(data)
    input_text = data['prompt']

    history = '\n'.join(conversation_history)

    final = f"{history}\nUser:{input_text}\nBot:"

    inputs = tokenizer.encode_plus(final,return_tensors = 'pt')
    outputs = model.generate(**inputs,do_sample = True, temperature = 0.7 , top_k = 50, top_p = 0.9)
    response = tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

    conversation_history.append(f"User:{input_text}")
    conversation_history.append(f"Bot:{response}")


    return response



if __name__ == "__main__":
    app.run()



