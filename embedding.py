from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
import json
import os

class Search:
    def __init__(self, art_df):
        """
        Search 클래스 초기화
        
        :param art_df: 작품 정보가 담긴 DataFrame
        """
        # 임베딩이 존재하지 않는 경우에만 인덱싱
        if os.path.exists("./chroma_db"):
            print("이미 존재하는 DB를 사용합니다")
            vectorstore_full = Chroma(
                persist_directory="./chroma_db", 
                collection_name="full_art_index",
                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large")
            )
    
            self.retriever_full = vectorstore_full.as_retriever(search_kwargs={'k': 6})
        else:
            print("새로운 임베딩을 생성합니다")
            # 작품 설명 임베딩
            self.retriever_full = self.art_embedding("full_art_index", art_df, art_df['작품 설명'].tolist(), topk=6)

    def art_embedding(self, collection_name, df, collection, topk):
        """
        작품 정보를 임베딩하고 검색기를 생성하는 메서드
        
        :param collection_name: 컬렉션 이름
        :param df: 작품 정보가 담긴 DataFrame
        :param collection: 임베딩할 텍스트 컬렉션
        :param topk: 검색 시 반환할 상위 결과 수
        :return: 생성된 검색기
        """
        art_descriptions = collection
        indexs = df['번호'].tolist()
        titles = df['유물명'].tolist()
        years = df['시대'].tolist()

        # Document 객체 생성
        docs = [
            Document(
                page_content=description,
                metadata={"index": index, "title": title, "year": year}
            )
            for description, index, title, year in zip(art_descriptions, indexs, titles, years)
        ]

        # Chroma를 사용하여 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        # 검색기 생성
        retriever = vectorstore.as_retriever(search_kwargs={'k': topk})
        
        return retriever

    def search(self, query, retrieval):
        """
        주어진 쿼리로 작품을 검색하는 메서드
        
        :param query: 검색 쿼리
        :param retrieval: 사용할 검색기
        :return: 검색 결과
        """
        docs = retrieval.invoke(query)
        return docs

    # BM25Retriever를 사용한 검색 기능 (현재 사용되지 않음)
    # def bm_search(self, query, retrieval):
    #     mecab = MeCab()
    #     query = " ".join(mecab.morphs(query))
    #     docs = retrieval.invoke(query)
    #     return docs