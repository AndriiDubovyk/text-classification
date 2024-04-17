import os

import PyPDF2
import docx
import pandas as pd


class DocumentParser:
    """A class for parsing text from .txt, .pdf, and .docx files.

    Methods:
        parse_to_text(file_path: str) -> str:
            Parses the document file specified by the given file path and returns its plain text content.
    """

    @staticmethod
    def _parse_pdf(file_path):
        """Parse the content of a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The plain text content of the PDF file.
        """
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text

    @staticmethod
    def _parse_docx(file_path):
        """Parse the content of a DOCX file.

        Args:
            file_path (str): The path to the DOCX file.

        Returns:
            str: The plain text content of the DOCX file.
        """
        doc = docx.Document(file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text

    @staticmethod
    def parse_to_text(file_path):
        """Parse the document file specified by the given file path and return its plain text content.

        Args:
            file_path (str): The path to the document file.

        Returns:
            str: The plain text content of the document file.

        Raises:
            ValueError: If the file format is unsupported.
        """
        if file_path.endswith('.pdf'):
            return DocumentParser._parse_pdf(file_path)
        elif file_path.endswith('.docx'):
            return DocumentParser._parse_docx(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def parse_files_to_df(directory):
        classes = []
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                classes.append(d)
        data = []
        for category in classes:
            category_dir = os.path.join(directory, category)
            for filename in os.listdir(category_dir):
                file_path = os.path.join(category_dir, filename)
                text = DocumentParser.parse_to_text(file_path)
                data.append({'label': category, 'text': text})
        return pd.DataFrame(data)
