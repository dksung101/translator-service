from mock import patch
from src.translator import translate_content, client
from sentence_transformers import SentenceTransformer, util
from typing import Callable
model = SentenceTransformer('all-MiniLM-L6-v2')

non_english_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "This is your first example.")
    },
    {
        "post": "Je viens de terminer mon projet.",
        "expected_answer": (False, "I just finished my project.")
    },
    {
        "post": "Món ăn yêu thích của tôi là pizza.",
        "expected_answer": (False, "My favorite food is pizza.")
    },
    {
        "post": "Le persone come te possono notare e prevenire algoritmi dannosi.",
        "expected_answer": (False, "People like you can notice and prevent harmful algorithms.")
    }
#     {
#         "post": "Media sosial hanya menggunakan suka dan komentar saya untuk menyarankan postingan.",
#         "expected_answer": (False, "Social media only uses my likes and comments to suggest posts.")
#     },
#     {
#         "post": "Sehemu salama za kuosha vyombo hurahisisha kusafisha.",
#         "expected_answer": (False, "Dishwasher safe parts makes clean-up easy.")
#     },
#     {
#         "post": "주말 산행을 마치고 돌아왔습니다! 전망은 숨이 막힐 정도로 아름다웠고, 공기는 ​​매우 신선했습니다. 자연에 둘러싸여 있으면 모든 것을 관점에서 볼 수 있는 뭔가가 있습니다. 기회가 된다면 자연 탈출을 즐겨보세요. 이는 정신 건강의 판도를 바꾸는 일입니다.",
#         "expected_answer": (False, "I’m back from my weekend getaway to the mountains! The views were breathtaking, and the air felt so fresh. There’s something about being surrounded by nature that puts everything into perspective. If you get a chance, treat yourself to a nature escape – it’s a game-changer for your mental health.")
#     },
#     {
#         "post": "Τον τελευταίο καιρό, ασχολούμαι με τη φωτογραφία και μου άνοιξε έναν εντελώς νέο κόσμο δημιουργικότητας. Η αποτύπωση της κατάλληλης στιγμής, το παιχνίδι του φωτός και των σκιών και ο τρόπος με τον οποίο μια εικόνα μπορεί να πει μια ιστορία χωρίς λόγια είναι μαγική. Κάποιες συμβουλές από συναδέλφους φωτογράφους εκεί έξω; Ψάχνετε πάντα να μαθαίνετε και να μεγαλώνετε!",
#         "expected_answer": (False, "Lately, I’ve been diving into photography, and it’s opened up a whole new world of creativity for me. Capturing the right moment, the play of light and shadows, and the way an image can tell a story without words is magical. Any tips from fellow photographers out there? Always looking to learn and grow!")
#     },
#     {
#         "post": "Mới nhận nuôi một chú chó con và tôi đã yêu rồi. Hãy gặp Max, quả cầu năng lượng nhỏ mịn nhất mà bạn từng thấy!",
#         "expected_answer": (False, "Just adopted a puppy, and I’m already in love. Meet Max, the fluffiest little ball of energy you’ve ever seen!")
#     },
#     {
#         "post": "ついにダウンタウンの新しい寿司屋を試してみました。味は最高でしたが、待ち時間が少し長すぎました。また行きますか？数時間の余裕があり、新鮮な刺身を食べたいという欲求があればかもしれません。",
#         "expected_answer": (False, "Finally tried that new sushi place downtown. The flavors were amazing, but the wait time was a bit too long. Would I go again? Maybe, if I had a few hours to spare and a craving for fresh sashimi.")
#     },
#     {
#         "post": "Es curioso cómo funciona la vida a veces. Encontré un viejo diario de hace cinco años y leer mis metas y sueños de entonces me hizo darme cuenta de cuánto he crecido. Algunas de las aspiraciones que tenía siguen siendo relevantes, pero otras me parecen muy alejadas de lo que soy ahora. Es un buen recordatorio de que todos evolucionamos constantemente y eso está perfectamente bien. Saludos por aceptar el cambio y esforzarnos por lograr las mejores versiones de nosotros mismos, incluso si parece diferente de lo que imaginamos.",
#         "expected_answer": (False, "It's funny how life works sometimes. I found an old journal from five years ago, and reading through my goals and dreams from back then made me realize how much I've grown. Some of the aspirations I had are still relevant, but others feel so far removed from who I am now. It's a good reminder that we're all constantly evolving, and that's perfectly okay. Cheers to embracing change and striving for the best versions of ourselves, even if it looks different than what we imagined.")
#     },
#     {
#         "post": "Только что вышел новый сезон моего любимого сериала, и я не могу перестать его смотреть! Повороты сюжета безумны, а развитие персонажей находится на другом уровне. Никаких спойлеров, но если вы еще не начали смотреть, вы пропускаете дикую поездку.",
#         "expected_answer": (False, "The new season of my favorite show just dropped, and I can’t stop binge-watching! The plot twists are insane, and the character development is on another level. No spoilers here, but if you haven’t started watching yet, you’re missing out on a wild ride.")
#     },
#     {
#         "post": "Tapaaminen oli onnistunut ja kaikki olivat yhtä mieltä pääkohdista.",
#         "expected_answer": (False, "The meeting was a success and everyone agreed on the main points.")
#     },
#     {
#         "post": "火车站在哪里？我迷路了，需要帮助。",
#         "expected_answer": (False, "Where is the train station? I am lost and need help.")
#     },
#     {
#         "post": "Som en, der har arbejdet hjemmefra i over tre år, har jeg bemærket, hvor vigtigt det er at opretholde en rutine. Det er nemt at lade grænserne mellem arbejde og privatliv udviske, men det at sætte klare grænser har gjort hele forskellen. Jeg har et dedikeret arbejdsområde, holder planlagte pauser og sørger for at \"logge af\" på et konsekvent tidspunkt. Og lad være med at få mig i gang med fordelene ved at have en frokostpause, der IKKE er foran computeren. For alle nye til fjernarbejde, giv dig selv nåde, mens du tilpasser dig. Det er en læreproces!",
#         "expected_answer": (False, "As someone who’s been working from home for over three years, I’ve noticed how important it is to maintain a routine. It’s easy to let the lines between work and personal life blur, but setting clear boundaries has made all the difference. I have a dedicated workspace, take scheduled breaks, and make sure to “log off” at a consistent time. And don’t get me started on the benefits of having a lunch break that’s NOT in front of the computer. For anyone new to remote work, give yourself grace as you adapt. It’s a learning process!")
#     }
]

english_eval_set = [
    {
        "post": "Coffee in hand, sunrise on the horizon, and a whole day ahead. Here’s to making the most of it!",
        "expected_answer": (True, "Coffee in hand, sunrise on the horizon, and a whole day ahead. Here’s to making the most of it!")
    },
    {
        "post": "Can’t believe I just finished reading a 600-page book in two days! Now I’m left with a book hangover and need recommendations.",
        "expected_answer": (True, "Can’t believe I just finished reading a 600-page book in two days! Now I’m left with a book hangover and need recommendations.")
    },
    {
        "post": "Just wrapped up a virtual cooking class, and it was so much fun! I learned to make authentic Italian pasta from scratch – turns out, it’s easier than I thought (and way tastier). Highly recommend trying something new like this, especially with friends. Nothing beats homemade!",
        "expected_answer": (True, "Just wrapped up a virtual cooking class, and it was so much fun! I learned to make authentic Italian pasta from scratch – turns out, it’s easier than I thought (and way tastier). Highly recommend trying something new like this, especially with friends. Nothing beats homemade!")
    },
    {
        "post": "Today, I learned that sometimes saying “no” is the best thing you can do for yourself. We live in a world that encourages us to say “yes” to every opportunity, every social event, every project. But sometimes, that “yes” can stretch us too thin. By saying “no” when I need to, I’m learning to honor my own time and energy. It’s okay not to please everyone, and it’s okay to choose rest over productivity. Self-care isn’t selfish – it’s essential. Here’s to setting boundaries and making room for what truly matters.",
        "expected_answer": (True, "Today, I learned that sometimes saying “no” is the best thing you can do for yourself. We live in a world that encourages us to say “yes” to every opportunity, every social event, every project. But sometimes, that “yes” can stretch us too thin. By saying “no” when I need to, I’m learning to honor my own time and energy. It’s okay not to please everyone, and it’s okay to choose rest over productivity. Self-care isn’t selfish – it’s essential. Here’s to setting boundaries and making room for what truly matters.")
    }
    # {
    #     "post": "Rainy days + a cup of tea + a good book = pure bliss.",
    #     "expected_answer": (True, "Rainy days + a cup of tea + a good book = pure bliss.")
    # },
    # {
    #     "post": "Tried learning the guitar during quarantine, and now I can finally play a full song without messing up! It’s amazing how a little progress each day can add up. Here’s to small wins and the patience to keep going!",
    #     "expected_answer": (True, "Tried learning the guitar during quarantine, and now I can finally play a full song without messing up! It’s amazing how a little progress each day can add up. Here’s to small wins and the patience to keep going!")
    # },
    # {
    #     "post": "After years of working in a corporate job, I finally made the leap to pursue my passion for writing full-time. It was one of the hardest decisions I’ve ever made, filled with doubts, fears, and what-ifs. But now, waking up every day knowing I’m doing what I love brings me a sense of fulfillment I didn’t even know I was missing. To anyone feeling stuck in a job that doesn’t fulfill them: it’s okay to take the leap when you’re ready. Trust yourself. It’s scary, but so worth it.",
    #     "expected_answer": (True, "After years of working in a corporate job, I finally made the leap to pursue my passion for writing full-time. It was one of the hardest decisions I’ve ever made, filled with doubts, fears, and what-ifs. But now, waking up every day knowing I’m doing what I love brings me a sense of fulfillment I didn’t even know I was missing. To anyone feeling stuck in a job that doesn’t fulfill them: it’s okay to take the leap when you’re ready. Trust yourself. It’s scary, but so worth it.")
    # },
    # {
    #     "post": "Just discovered my new favorite dessert: chocolate lava cake. How did I live this long without it?!",
    #     "expected_answer": (True, "Just discovered my new favorite dessert: chocolate lava cake. How did I live this long without it?!")
    # },
    # {
    #     "post": "Went to a farmer’s market for the first time in ages, and wow, I forgot how great it feels to buy fresh produce directly from local farmers. Got everything I need for a big veggie stir-fry tonight. There’s something about knowing exactly where your food comes from that makes it taste so much better!",
    #     "expected_answer": (True, "Went to a farmer’s market for the first time in ages, and wow, I forgot how great it feels to buy fresh produce directly from local farmers. Got everything I need for a big veggie stir-fry tonight. There’s something about knowing exactly where your food comes from that makes it taste so much better!")
    # },
    # {
    #     "post": "Traveling alone was something I always thought I’d be too nervous to try, but this past week, I finally did it. I explored new cities, met amazing people, and learned so much about myself. I realized how much I value independence and the freedom to set my own pace. Solo travel isn’t always easy, but it’s deeply rewarding. It teaches you resilience, self-reliance, and gives you a fresh perspective on the world. If anyone’s considering a solo trip, I say go for it! You’ll come back with more than just memories – you’ll come back a bit stronger.",
    #     "expected_answer": (True, "Traveling alone was something I always thought I’d be too nervous to try, but this past week, I finally did it. I explored new cities, met amazing people, and learned so much about myself. I realized how much I value independence and the freedom to set my own pace. Solo travel isn’t always easy, but it’s deeply rewarding. It teaches you resilience, self-reliance, and gives you a fresh perspective on the world. If anyone’s considering a solo trip, I say go for it! You’ll come back with more than just memories – you’ll come back a bit stronger.")
    # },
    # {
    #     "post": "Went stargazing last night, and I’m still in awe of how vast the universe is. It’s humbling, really.",
    #     "expected_answer": (True, "Went stargazing last night, and I’m still in awe of how vast the universe is. It’s humbling, really.")
    # },
    # {
    #     "post": "Why do the best ideas always come to me at 3 a.m.? Sleep schedule: ruined. Creativity: thriving.",
    #     "expected_answer": (True, "Why do the best ideas always come to me at 3 a.m.? Sleep schedule: ruined. Creativity: thriving.")
    # },
    # {
    #     "post": "Tried out a new workout routine today, and wow, my muscles are on fire! It’s tough, but it feels so good to push my limits. Here’s to the soreness that means progress.",
    #     "expected_answer": (True, "Tried out a new workout routine today, and wow, my muscles are on fire! It’s tough, but it feels so good to push my limits. Here’s to the soreness that means progress.")
    # },
    # {
    #     "post": "I’ve been doing a lot of reflecting on friendships lately. It’s interesting how some people come into your life for a short time, but they make a big impact. And then there are the friends who stick around through every phase, no matter how much you change. It reminds me that quality over quantity is what truly matters. To anyone out there feeling lonely – keep putting yourself out there. Real connections are worth the wait, and they might just surprise you when you least expect it.",
    #     "expected_answer": (True, "I’ve been doing a lot of reflecting on friendships lately. It’s interesting how some people come into your life for a short time, but they make a big impact. And then there are the friends who stick around through every phase, no matter how much you change. It reminds me that quality over quantity is what truly matters. To anyone out there feeling lonely – keep putting yourself out there. Real connections are worth the wait, and they might just surprise you when you least expect it.")
    # },
    # {
    #     "post": "Just tried the new pumpkin spice latte, and I get the hype now.",
    #     "expected_answer": (True, "Just tried the new pumpkin spice latte, and I get the hype now.")
    # }
]

gibberish_eval_set = [
    {
        "post": "asdfghjkjkslswert",
        "expected_answer": (True, "Error processing post")
    },
    {
        "post": "dasdghqwiro sdaf pqetkglds",
        "expected_answer": (True, "Error processing post")
    },
    {
        "post": "sadf asdfa ri3359025 ",
        "expected_answer": (True, "Error processing post")
    },
    {
        "post": "asdf asdf aksdfjsadkfld",
        "expected_answer": (True, "Error processing post")
    },
    {
        "post": "sfdj;saldfkjf",
        "expected_answer": (True, "Error processing post")
    },
]

def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
  '''Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
  expected_embedding = model.encode(expected_answer)
  response_embedding = model.encode(llm_response)

  similarity = model.similarity(expected_embedding, response_embedding)
  return similarity.item()

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
  '''Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
  sentence_result = eval_single_response_translation(expected_answer[1], llm_response[1])
  bool_result = expected_answer[0] == llm_response[0]
  if bool_result:
    return (sentence_result + 1.0)/2
  return (sentence_result + 0.0)/2

def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
  '''
  Computes an aggregate score of the chosen evaluation metric across the given dataset. Calls the query_fn function to generate
  LLM outputs for each of the posts in the evaluation dataset, and calls eval_single_response to calculate the metric.
  '''
  sum = 0
  for post in dataset:
    llm_response = query_fn(post["post"])
    score = eval_fn(post["expected_answer"], llm_response)
    sum += score
  return sum/len(dataset)

def test_llm_translate_to_english_response():
    non_eng_eval_score = evaluate(translate_content, eval_single_response_complete, non_english_eval_set)
    assert non_eng_eval_score >= 0.90

def test_llm_detect_english_response():
    eng_eval_score = evaluate(translate_content, eval_single_response_complete, english_eval_set)
    assert eng_eval_score >= 0.90

def test_llm_gibberish_response():
    gibberish_eval_score = evaluate(translate_content, eval_single_response_complete, non_english_eval_set)
    assert gibberish_eval_score >= 0.60

@patch.object(client.chat.completions, 'create')
def test_unexpected_language(mocker):
  # we mock the model's response to return a random message
  mocker.return_value.choices[0].message.content = "I don't understand your request"

  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Hier ist dein erstes Beispiel.")

@patch.object(client.chat.completions, 'create')
def test_empty_response(mocker):
    # Mock the model's response to return an empty message
    mocker.return_value.choices[0].message.content = ""

    assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Hier ist dein erstes Beispiel.")

@patch.object(client.chat.completions, 'create')
def test_unexpected_bool_val(mocker):
    # Mock the model's response to return an message with a number other than 0 or 1
    mocker.return_value.choices[0].message.content = "2 This is your first example."

    assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Hier ist dein erstes Beispiel.")

@patch.object(client.chat.completions, 'create')
def test_empty_bool_val(mocker):
    # Mock the model's response to return an message with no number
    mocker.return_value.choices[0].message.content = "This is your first example."

    assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Hier ist dein erstes Beispiel.")

@patch.object(client.chat.completions, 'create')
def test_no_separation(mocker):
    # Mock the model's response to return a message not formatted correctly
    mocker.return_value.choices[0].message.content = "0This is your first example."

    assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Hier ist dein erstes Beispiel.")