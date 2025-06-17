from typing import List, Optional
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json
from quick_pp.logger import logger


class DocumentProcessor:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="qwen3")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_file(self, file_path: str) -> List[Document]:
        """Process a file and return a list of Document objects."""
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == '.pdf':
                return self._process_pdf(file_path)
            elif file_extension == '.docx':
                return self._process_docx(file_path)
            elif file_extension == '.txt':
                return self._process_txt(file_path)
            elif file_extension == '.csv':
                return self._process_csv(file_path)
            elif file_extension == '.json':
                return self._process_json(file_path)
            else:
                # Use UnstructuredLoader for other file types
                return self._process_unstructured(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files using PyMuPDFLoader."""
        try:
            # First try PyMuPDFLoader for better text extraction
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            # If no text was extracted, try UnstructuredLoader with OCR
            if not documents or not any(doc.page_content.strip() for doc in documents):
                logger.info(f"No text extracted with PyMuPDFLoader, trying UnstructuredLoader with OCR for {file_path}")
                return self._process_unstructured(file_path, use_ocr=True)

            return documents
        except Exception as e:
            logger.error(f"Error in PyMuPDFLoader, falling back to UnstructuredLoader: {str(e)}")
            return self._process_unstructured(file_path)

    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files using UnstructuredLoader."""
        return self._process_unstructured(file_path)

    def _process_txt(self, file_path: str) -> List[Document]:
        """Process TXT files using UnstructuredLoader."""
        return self._process_unstructured(file_path)

    def _process_csv(self, file_path: str) -> List[Document]:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            text = df.to_string()
            return [Document(
                page_content=text,
                metadata={"source": file_path, "type": "csv"}
            )]
        except Exception as e:
            logger.error(f"Error processing CSV, trying UnstructuredLoader: {str(e)}")
            return self._process_unstructured(file_path)

    def _process_json(self, file_path: str) -> List[Document]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            text = json.dumps(data, indent=2)
            return [Document(
                page_content=text,
                metadata={"source": file_path, "type": "json"}
            )]
        except Exception as e:
            logger.error(f"Error processing JSON, trying UnstructuredLoader: {str(e)}")
            return self._process_unstructured(file_path)

    def _process_unstructured(self, file_path: str, use_ocr: bool = False) -> List[Document]:
        """Process files using UnstructuredLoader with optional OCR."""
        try:
            if use_ocr:
                # Use RapidOCR for better text extraction from images
                parser = RapidOCRBlobParser()
                loader = UnstructuredLoader(file_path, parser=parser)
            else:
                loader = UnstructuredLoader(file_path)

            documents = loader.load()

            # Add metadata about the processing method
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "processor": "unstructured",
                    "ocr_used": use_ocr
                })

            return documents
        except Exception as e:
            logger.error(f"Error in UnstructuredLoader: {str(e)}", exc_info=True)
            raise

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create a vector store from documents."""
        try:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)

            # Create and persist vector store
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
            raise

    def load_vectorstore(self) -> Optional[Chroma]:
        """Load an existing vector store."""
        try:
            if os.path.exists(self.persist_directory):
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            return None
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            raise
