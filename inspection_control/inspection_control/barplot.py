import matplotlib.pyplot as plt

methods = ["Nakhaeinia", "Proposed"]
final_error = [0.09, 0.00684]

plt.bar(methods, final_error)
plt.ylabel("Final Orientation Error (rad)")
plt.title("Final Error Comparison")

# Optional (VERY powerful visually)
plt.yscale('log')

plt.show()