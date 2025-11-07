import matplotlib.pyplot as plt

# Data
classes = ['Real', 'Fake']
precision = [0.64, 0.87]

# Create the plot
plt.figure(figsize=(6, 5))
bars = plt.bar(classes, precision, color=['skyblue', 'salmon'])

# Add value labels on top of each bar
for bar, value in zip(bars, precision):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.2f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels and title
plt.ylim(0, 1.05)
plt.ylabel('Precision', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.title('XGBoost Precision by Class', fontsize=14, fontweight='bold')
plt.savefig("precision_chart.png", dpi=300)


# Show the plot
plt.tight_layout()
plt.show()
