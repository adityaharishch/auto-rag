import matplotlib.pyplot as plt
import pandas as pd

data = {
    'AiScore': [1, 2, 3, 4],
    '66fc6f6f-5663-470e-8206-55a0464d884a': [5, 4, 7, 25],
    'dd29ac00-2094-48a6-b9ae-63e14d426b5e': [2, 10, 18, 19]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots()
df.plot(x='AiScore', kind='bar', ax=ax)

ax.set_xlabel('AiScore')
ax.set_ylabel('Number of Students')
ax.set_title('Number of Students for each AiScore by QuestionId')
ax.legend(title='QuestionId')

# Save plot
plt.savefig('ai_score_bar_graph.png')
plt.show()