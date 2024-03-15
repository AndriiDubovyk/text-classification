import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Combobox

from document_parser import DocumentParser
from text_classifier_simple import TextClassifier


class TextClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Text Classifier")
        master.geometry("400x600")

        self.languages_label = tk.Label(master, text="Select Language:", font=("Arial", 12))
        self.languages_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)

        self.languages_combobox = Combobox(master, values=["English", "Ukrainian"], state="readonly",
                                           font=("Arial", 12))
        self.languages_combobox.current(0)  # Set default language to English
        self.languages_combobox.grid(row=0, column=1, columnspan=2, sticky=tk.W + tk.E, padx=10, pady=5)

        self.categories_label = tk.Label(master, text="Enter Categories:", font=("Arial", 12))
        self.categories_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)

        self.categories_entry = tk.Entry(master, font=("Arial", 12))
        self.categories_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W + tk.E, padx=10, pady=5)

        self.select_files_button = tk.Button(master, text="Select Files", command=self.select_files, font=("Arial", 12))
        self.select_files_button.grid(row=2, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

        self.files_label = tk.Label(master, text="Selected Files:", font=("Arial", 12))
        self.files_label.grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)

        self.files_listbox = tk.Listbox(master, width=40, height=10, font=("Arial", 12))
        self.files_listbox.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W + tk.E)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit, font=("Arial", 12))
        self.submit_button.grid(row=5, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

        self.copy_button = tk.Button(master, text="Copy Results", command=self.copy_results, state=tk.DISABLED,
                                     font=("Arial", 12))
        self.copy_button.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

        self.results_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10, font=("Arial", 12))
        self.results_text.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W + tk.E)

        self.file_paths = []

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)
        master.grid_rowconfigure(4, weight=1)
        master.grid_rowconfigure(7, weight=1)

    def select_files(self):
        files = filedialog.askopenfilenames()
        if files:
            self.file_paths = files
            self.files_listbox.delete(0, tk.END)
            for file_path in self.file_paths:
                self.files_listbox.insert(tk.END, file_path)

    def submit(self):
        language = self.languages_combobox.get().lower()
        categories = self.categories_entry.get().split(',')
        if not categories:
            messagebox.showerror("Error", "Please enter categories.")
            return

        if not self.file_paths:
            messagebox.showerror("Error", "No files selected.")
            return

        classifier = TextClassifier(language=language)
        texts = [DocumentParser.parse_to_text(file_path) for file_path in self.file_paths]
        predicted_categories = classifier.classify(categories, texts)

        self.results_text.delete(1.0, tk.END)
        for file, category in zip(self.file_paths, predicted_categories):
            self.results_text.insert(tk.END, f"{file} - {category}\n")

        self.copy_button.config(state=tk.NORMAL)

    def copy_results(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.results_text.get(1.0, tk.END))

        messagebox.showinfo("Copied", "Results copied to clipboard.")


def main():
    root = tk.Tk()
    TextClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
