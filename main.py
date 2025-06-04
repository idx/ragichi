import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import time

# Log configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Search result document"""
    content: str
    metadata: Dict[str, Any] = None

class OllamaAPIClient:
    """Ollama API client"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    async def generate(self, model: str, prompt: str, system: str = "", temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
        """Generate text with Ollama API and return response with stats"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        inference_time = time.time() - start_time
                        
                        stats = {
                            "inference_time": inference_time,
                            "prompt_eval_count": result.get("prompt_eval_count", 0),
                            "eval_count": result.get("eval_count", 0),
                            "total_duration": result.get("total_duration", 0) / 1e9 if result.get("total_duration") else 0,  # Convert from nanoseconds
                            "load_duration": result.get("load_duration", 0) / 1e9 if result.get("load_duration") else 0,
                            "prompt_eval_duration": result.get("prompt_eval_duration", 0) / 1e9 if result.get("prompt_eval_duration") else 0,
                            "eval_duration": result.get("eval_duration", 0) / 1e9 if result.get("eval_duration") else 0
                        }
                        
                        return result.get("response", ""), stats
                    else:
                        error_text = await response.text()
                        raise Exception(f"APIエラー {response.status}: {error_text}")
            except Exception as e:
                logger.error(f"Ollama APIエラー: {e}")
                raise

class MockVectorStore:
    """Mock vector store (actual RAG systems use real vector DB)"""
    
    def __init__(self):
        # Sample documents
        self.documents = [
            Document("Pythonはシンプルで読みやすい構文を特徴とする高水準プログラミング言語です。"),
            Document("機械学習ライブラリには、scikit-learn、TensorFlow、PyTorchなどがあります。"),
            Document("FastAPIは自動ドキュメント生成機能を持つ高性能なWeb APIフレームワークです。"),
            Document("非同期プログラミングでは、asyncio、aiohttp、aiodataなどのライブラリを使用します。"),
            Document("データ分析では、pandas、NumPy、matplotlibなどのライブラリが広く使われています。"),
        ]
    
    async def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Search documents based on query (mock implementation)"""
        # In actual systems, similarity search with embedding vectors is performed
        logger.info(f"ベクトル検索を実行中: {query}")
        
        # Simple keyword matching (actually embedding search)
        results = []
        for doc in self.documents:
            if any(word.lower() in doc.content.lower() for word in query.split()):
                results.append(doc)
        
        return results[:top_k]

class RAGChainSystem:
    """RAG chain system"""
    
    def __init__(self, ollama_client: OllamaAPIClient, vector_store: MockVectorStore, model: str = "llama3"):
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.model = model
        self.stats = {}  # Store stats for each LLM call
        
        # System prompt for query rewriting
        self.rewrite_system_prompt = """あなたは検索クエリ最適化の専門家です。
ユーザーの質問を分析し、より効果的な検索結果を得るために最適化されたクエリに再構成してください。

タスク:
1. ユーザーの意図を理解する
2. 重要なキーワードを特定する
3. 検索に適した形式に再構成する
4. 複数の検索角度がある場合は、最も重要なものを選択する

出力形式: 最適化されたクエリのみを出力してください（説明は不要）"""

        # System prompt for RAG generation
        self.generate_system_prompt = """あなたは提供されたコンテキスト情報に基づいて質問に答えるアシスタントです。

重要な指示:
1. 回答には提供されたコンテキスト情報のみを使用してください
2. コンテキストに情報がない場合は「提供された情報では回答できません」と述べてください
3. 正確かつ簡潔な回答を心がけてください
4. 推測や想像は避け、事実に基づいた回答のみを提供してください
5. 適切な言語で自然な応答を提供してください"""

    async def rewrite_query(self, original_query: str) -> str:
        """Step 1: Query rewriting"""
        logger.info(f"クエリ書き換えを開始: {original_query}")
        
        prompt = f"ユーザーの質問: {original_query}"
        
        rewritten_query, stats = await self.ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=self.rewrite_system_prompt,
            temperature=0.3
        )
        
        self.stats['rewrite'] = stats
        rewritten_query = rewritten_query.strip()
        logger.info(f"書き換え後のクエリ: {rewritten_query}")
        return rewritten_query

    async def retrieve_documents(self, query: str) -> List[Document]:
        """Step 2: Document retrieval"""
        logger.info(f"ドキュメント取得を開始: {query}")
        
        documents = await self.vector_store.search(query, top_k=3)
        logger.info(f"検索結果: {len(documents)}件のドキュメント")
        
        return documents

    async def generate_response(self, original_query: str, documents: List[Document]) -> str:
        """Step 3: RAG generation"""
        logger.info("応答生成を開始")
        
        # Build context
        context = "\n\n".join([f"ドキュメント {i+1}: {doc.content}" for i, doc in enumerate(documents)])
        
        prompt = f"""コンテキスト情報:
{context}

ユーザーの質問: {original_query}

上記のコンテキスト情報に基づいて質問に答えてください。"""

        response, stats = await self.ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=self.generate_system_prompt,
            temperature=0.5
        )
        
        self.stats['generate'] = stats
        logger.info("応答生成が完了")
        return response.strip()

    async def process(self, user_query: str) -> Dict[str, Any]:
        """Main process: Query rewriting → Retrieval → Generation"""
        logger.info(f"RAGチェーンプロセスを開始: {user_query}")
        
        # Reset stats for new process
        self.stats = {}
        
        try:
            # Step 1: Query rewriting
            rewritten_query = await self.rewrite_query(user_query)
            
            # Step 2: Document retrieval
            documents = await self.retrieve_documents(rewritten_query)
            
            # Step 3: Response generation
            final_response = await self.generate_response(user_query, documents)
            
            result = {
                "original_query": user_query,
                "rewritten_query": rewritten_query,
                "retrieved_documents": [doc.content for doc in documents],
                "final_response": final_response,
                "stats": self.stats,
                "success": True
            }
            
            logger.info("RAGチェーンプロセスが完了")
            return result
            
        except Exception as e:
            logger.error(f"RAGチェーンプロセスエラー: {e}")
            return {
                "original_query": user_query,
                "error": str(e),
                "success": False
            }

async def display_result(result: Dict[str, Any]) -> None:
    """Display the RAG result in a formatted way"""
    if result["success"]:
        print(f"\n{'='*60}")
        print("結果")
        print('='*60)
        
        print(f"\n書き換え後のクエリ:")
        print(f"  {result['rewritten_query']}")
        
        print(f"\n取得したドキュメント:")
        for i, doc in enumerate(result["retrieved_documents"], 1):
            print(f"  {i}. {doc}")
        
        print(f"\n最終回答:")
        print("-" * 40)
        print(result["final_response"])
        print("-" * 40)
        
        # Display LLM stats
        if "stats" in result:
            print(f"\n{'='*60}")
            print("LLM推論統計")
            print('='*60)
            
            if "rewrite" in result["stats"]:
                stats = result["stats"]["rewrite"]
                print(f"\nクエリ書き換え:")
                print(f"  - 入力トークン数: {stats['prompt_eval_count']}")
                print(f"  - 出力トークン数: {stats['eval_count']}")
                print(f"  - 合計トークン数: {stats['prompt_eval_count'] + stats['eval_count']}")
                print(f"  - 推論時間: {stats['inference_time']:.2f}秒")
                print(f"  - プロンプト評価時間: {stats['prompt_eval_duration']:.2f}秒")
                print(f"  - 生成時間: {stats['eval_duration']:.2f}秒")
            
            if "generate" in result["stats"]:
                stats = result["stats"]["generate"]
                print(f"\n応答生成:")
                print(f"  - 入力トークン数: {stats['prompt_eval_count']}")
                print(f"  - 出力トークン数: {stats['eval_count']}")
                print(f"  - 合計トークン数: {stats['prompt_eval_count'] + stats['eval_count']}")
                print(f"  - 推論時間: {stats['inference_time']:.2f}秒")
                print(f"  - プロンプト評価時間: {stats['prompt_eval_duration']:.2f}秒")
                print(f"  - 生成時間: {stats['eval_duration']:.2f}秒")
            
            # Total stats
            total_input_tokens = sum(s.get('prompt_eval_count', 0) for s in result["stats"].values())
            total_output_tokens = sum(s.get('eval_count', 0) for s in result["stats"].values())
            total_inference_time = sum(s.get('inference_time', 0) for s in result["stats"].values())
            
            print(f"\n合計:")
            print(f"  - 総入力トークン数: {total_input_tokens}")
            print(f"  - 総出力トークン数: {total_output_tokens}")
            print(f"  - 総トークン数: {total_input_tokens + total_output_tokens}")
            print(f"  - 総推論時間: {total_inference_time:.2f}秒")
    else:
        print(f"\nエラー: {result['error']}")
        print("Ollamaサーバーが実行中で、モデルがインストールされていることを確認してください。")


async def run_cli():
    """Run the interactive CLI"""
    print("RAGチェーンシステム - インタラクティブCLI")
    print("="*60)
    print("RAG（Retrieval-Augmented Generation）システムへようこそ！")
    print("\nナレッジベース内の利用可能なドキュメント:")
    
    # Show available documents
    vector_store = MockVectorStore()
    for i, doc in enumerate(vector_store.documents, 1):
        print(f"  {i}. {doc.content[:60]}...")
    
    print("\n注意: Ollamaサーバーがhttp://localhost:11434で実行されている必要があります")
    print("プログラムを終了するには'quit'または'exit'と入力してください")
    print("="*60)
    
    # Initialize system
    try:
        ollama_client = OllamaAPIClient()
        rag_system = RAGChainSystem(ollama_client, vector_store, model="gemma3:4b")
        
        while True:
            try:
                # Get user input
                print("\n質問を入力してください:")
                user_query = input("> ").strip()
                
                # Check for exit commands
                if user_query.lower() in ['quit', 'exit']:
                    print("\nRAGシステムをご利用いただきありがとうございました。さようなら！")
                    break
                
                # Skip empty queries
                if not user_query:
                    print("有効な質問を入力してください。")
                    continue
                
                # Process the query
                print("\nクエリを処理中...")
                result = await rag_system.process(user_query)
                
                # Display results
                await display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n操作がキャンセルされました。終了するには'quit'と入力してください。")
                continue
            except Exception as e:
                logger.error(f"クエリ処理エラー: {e}")
                print(f"\nエラーが発生しました: {e}")
                print("再度試すか、Ollamaサーバーの接続を確認してください。")
    
    except Exception as e:
        print(f"\nRAGシステムの初期化に失敗しました: {e}")
        print("\n以下を確認してください:")
        print("  1. Ollamaサーバーが実行中: ollama serve")
        print("  2. 必要なモデルがインストールされている: ollama pull llama3")


# Main entry point
if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n\nプログラムが中断されました。さようなら！")
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        logger.error(f"致命的エラー: {e}", exc_info=True)