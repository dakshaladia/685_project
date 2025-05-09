# tgp_v1__qXtXIr9Y_EjIAJQd_CaI_ibEkdZp8874dzvUQUGgeU



from together import Together

client = Together()

stream = client.chat.completions.create(
  model="Qwen/Qwen2.5-7B-Instruct-Turbo",
  messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
  stream=True,
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="", flush=True)