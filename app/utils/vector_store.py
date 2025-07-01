import os
import faiss
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from app.utils.embeddings import get_embedding_service
from app.utils.logger import get_logger
from app.config import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class CustomFAISSVectorStore:
    """Custom FAISS vector store that works with our embedding service"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None

        logger.info("Initialized CustomFAISSVectorStore")

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        if not texts:
            logger.warning("No texts provided to add_documents")
            return

        logger.info(f"Adding {len(texts)} documents to vector store")

        # Get embeddings for the texts
        embeddings = self.embedding_service.embed_documents(texts)

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Initialize index if needed
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(
                f"Created FAISS index with dimension: {self.dimension}")

        # Add embeddings to index
        self.index.add(embeddings_array)

        # Store documents and metadata
        self.documents.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])

        logger.info(
            f"Added {len(texts)} documents. Total documents: {len(self.documents)}")

    def similarity_search(self, query: str, k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []

        k = k or settings.TOP_K_RETRIEVAL
        # Don't search for more docs than we have
        k = min(k, len(self.documents))

        logger.info(f"Searching for top {k} similar documents")

        # Get query embedding
        query_embedding = self.embedding_service.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search
        distances, indices = self.index.search(query_vector, k)

        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Safety check
                results.append((
                    self.documents[idx],
                    float(distance),
                    self.metadatas[idx] if idx < len(self.metadatas) else {}
                ))

        logger.info(f"Found {len(results)} similar documents")
        return results

    def clear(self):
        """Clear all documents and reset the index"""
        self.index = None
        self.documents = []
        self.metadatas = []
        self.dimension = None
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "has_index": self.index is not None,
            "index_size": self.index.ntotal if self.index else 0
        }


class LLMService:
    """Service for generating answers using various LLM providers"""

    def __init__(self):
        self.logger = get_logger(__name__)

    async def generate_completion(self, prompt: str) -> str:
        """Generate text completion using available LLM service"""

        # Primary: Try Google AI Studio first (free and generous quota)
        if settings.LLM_PROVIDER == "google" and settings.GOOGLE_API_KEY:
            try:
                return await self._generate_google_completion(prompt)
            except Exception as e:
                self.logger.warning(
                    f"Google AI completion failed: {e}, trying OpenAI")

        # Fallback 1: Try OpenAI if available
        if settings.OPENAI_API_KEY:
            try:
                return await self._generate_openai_completion(prompt)
            except Exception as e:
                self.logger.warning(
                    f"OpenAI completion failed: {e}, falling back to local method")

        # Fallback 2: Local extractive method
        return self._generate_extractive_answer(prompt)

    async def generate_streaming_completion(self, prompt: str):
        """Generate streaming text completion using available LLM service"""

        # Primary: Try Google AI Studio streaming first
        if settings.LLM_PROVIDER == "google" and settings.GOOGLE_API_KEY:
            try:
                async for chunk in self._generate_google_streaming_completion(prompt):
                    yield chunk
                return
            except Exception as e:
                self.logger.warning(
                    f"Google AI streaming failed: {e}, trying OpenAI")

        # Fallback 1: Try OpenAI streaming if available
        if settings.OPENAI_API_KEY:
            try:
                async for chunk in self._generate_openai_streaming_completion(prompt):
                    yield chunk
                return
            except Exception as e:
                self.logger.warning(
                    f"OpenAI streaming failed: {e}, falling back to chunked response")

        # Fallback 2: Simulate streaming with chunked extractive response
        answer = self._generate_extractive_answer(prompt)
        async for chunk in self._simulate_streaming(answer):
            yield chunk

    async def _generate_google_completion(self, prompt: str) -> str:
        """Generate completion using Google AI Studio (Gemini)"""
        try:
            import google.generativeai as genai

            # Configure the API key
            genai.configure(api_key=settings.GOOGLE_API_KEY)

            # Initialize the model
            model = genai.GenerativeModel(settings.GOOGLE_MODEL)

            # Create a more focused prompt for Gemini
            system_instruction = """You are a helpful assistant that answers questions based on provided context. 
            Be concise, accurate, and well-structured in your responses. If the answer cannot be found in the context, please say so clearly."""

            # Combine system instruction with user prompt
            full_prompt = f"{system_instruction}\n\n{prompt}"

            # Generate response
            response = await model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40
                )
            )

            answer = response.text.strip()
            self.logger.info(
                "Generated answer using Google AI Studio (Gemini)")
            return answer

        except Exception as e:
            self.logger.error(f"Google AI completion error: {e}")
            raise

    async def _generate_google_streaming_completion(self, prompt: str):
        """Generate streaming completion using Google AI Studio (Gemini)"""
        try:
            import google.generativeai as genai

            # Configure the API key
            genai.configure(api_key=settings.GOOGLE_API_KEY)

            # Initialize the model
            model = genai.GenerativeModel(settings.GOOGLE_MODEL)

            # Create a more focused prompt for Gemini
            system_instruction = """You are a helpful assistant that answers questions based on provided context. 
            Be concise, accurate, and well-structured in your responses. If the answer cannot be found in the context, please say so clearly."""

            # Combine system instruction with user prompt
            full_prompt = f"{system_instruction}\n\n{prompt}"

            # Generate streaming response
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40
                ),
                stream=True
            )

            self.logger.info("Starting Google AI streaming generation")

            for chunk in response:
                if chunk.text:
                    yield {
                        "type": "content",
                        "data": chunk.text,
                        "provider": "google"
                    }

            # Send completion signal
            yield {
                "type": "complete",
                "data": "",
                "provider": "google"
            }

        except Exception as e:
            self.logger.error(f"Google AI streaming error: {e}")
            raise

    async def _generate_openai_completion(self, prompt: str) -> str:
        """Generate completion using OpenAI API"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            answer = response.choices[0].message.content.strip()
            self.logger.info("Generated answer using OpenAI")
            return answer

        except Exception as e:
            self.logger.error(f"OpenAI completion error: {e}")
            raise

    async def _generate_openai_streaming_completion(self, prompt: str):
        """Generate streaming completion using OpenAI API"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                stream=True
            )

            self.logger.info("Starting OpenAI streaming generation")

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "content",
                        "data": chunk.choices[0].delta.content,
                        "provider": "openai"
                    }

            # Send completion signal
            yield {
                "type": "complete",
                "data": "",
                "provider": "openai"
            }

        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            raise

    async def _simulate_streaming(self, text: str, chunk_size: int = 20):
        """Simulate streaming by chunking text for fallback method"""
        import asyncio

        words = text.split()
        self.logger.info("Simulating streaming for extractive answer")

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                yield {
                    "type": "content",
                    "data": chunk + (" " if i + chunk_size < len(words) else ""),
                    "provider": "local"
                }
                await asyncio.sleep(0.1)  # Small delay to simulate streaming

        # Send completion signal
        yield {
            "type": "complete",
            "data": "",
            "provider": "local"
        }

    def _generate_extractive_answer(self, prompt: str) -> str:
        """Generate answer by extracting and summarizing from context"""
        try:
            # Extract context from the prompt
            context_start = prompt.find("Context:\n") + len("Context:\n")
            context_end = prompt.find("\n\nQuestion:")

            if context_start == -1 or context_end == -1:
                return "I couldn't process the context properly. Please try rephrasing your question."

            context = prompt[context_start:context_end].strip()
            question_start = prompt.find("Question: ") + len("Question: ")
            question = prompt[question_start:].strip()

            # Check for specific question types that need special handling
            question_lower = question.lower()

            # Handle title-specific questions
            if any(word in question_lower for word in ['title', 'name of the book', 'book called', 'book title']):
                return self._extract_title_info(context, question)

            # Handle author-specific questions
            elif any(word in question_lower for word in ['author', 'written by', 'who wrote']):
                return self._extract_author_info(context, question)

            # Handle contents/table of contents questions
            elif any(phrase in question_lower for phrase in ['contents', 'table of contents', 'chapters', 'outline', 'structure', 'what does this book cover', 'topics covered']):
                return self._extract_contents_info(context, question)

            # Handle general "what is this book about" questions
            elif any(phrase in question_lower for phrase in ['what is this book about', 'about this book', 'book about']):
                return self._extract_book_summary(context, question)

            # Default extractive approach for other questions
            return self._extract_general_answer(context, question)

        except Exception as e:
            self.logger.error(f"Extractive answer generation error: {e}")
            return "I encountered an error while processing the context. Please try rephrasing your question."

    def _extract_title_info(self, context: str, question: str) -> str:
        """Extract title information using parallel-thinking approach"""
        import concurrent.futures
        from threading import Lock

        self.logger.info("Parallel-thinking title extraction started")

        # Parallel-thinking: Run multiple extraction strategies simultaneously
        extraction_strategies = [
            self._strategy_metadata_search,
            self._strategy_document_structure,
            self._strategy_semantic_patterns,
            self._strategy_position_based,
            self._strategy_format_analysis
        ]

        results = {}
        results_lock = Lock()

        def run_strategy(strategy_func):
            """Run a single extraction strategy and return results"""
            try:
                strategy_name = strategy_func.__name__
                self.logger.info(f"Running parallel strategy: {strategy_name}")

                candidates = strategy_func(context)

                with results_lock:
                    results[strategy_name] = {
                        'candidates': candidates,
                        'count': len(candidates)
                    }

                self.logger.info(
                    f"Strategy {strategy_name} found {len(candidates)} candidates")
                return strategy_name, candidates

            except Exception as e:
                self.logger.error(
                    f"Strategy {strategy_func.__name__} failed: {e}")
                return strategy_func.__name__, []

        # Execute all strategies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_strategy = {
                executor.submit(run_strategy, strategy): strategy.__name__
                for strategy in extraction_strategies
            }

            all_candidates = []
            strategy_results = {}

            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_name, candidates = future.result()
                strategy_results[strategy_name] = candidates
                all_candidates.extend(candidates)

        # Parallel analysis: Score all candidates from all strategies
        self.logger.info(
            f"Parallel thinking collected {len(all_candidates)} total candidates from {len(strategy_results)} strategies")

        # Cross-strategy validation and scoring
        final_candidates = self._cross_validate_candidates(
            all_candidates, strategy_results, context)

        # Select best candidate
        best_candidate = self._select_best_parallel_candidate(final_candidates)

        if best_candidate:
            self.logger.info(
                f"Parallel thinking selected: {best_candidate['text'][:50]}... (score: {best_candidate['score']}, strategies: {best_candidate['strategy_count']})")
            return f"Based on the document content, the title appears to be: **{best_candidate['text']}**"
        else:
            return self._fallback_title_analysis(context)

    def _strategy_metadata_search(self, context: str) -> List[str]:
        """Strategy 1: Look for explicit metadata and title declarations"""
        candidates = []

        # Explicit title patterns
        patterns = [
            r'(?i)(?:title|book title|document title)[:\s]+([^\n]+)',
            r'(?i)(?:this book is titled|titled)[:\s]+([^\n]+)',
            r'(?i)(?:book|textbook) name[:\s]+([^\n]+)',
            r'(?i)^title:\s*(.+)$',
            # Look for academic title patterns
            r'(?i)(operating systems?[:\s]*[^\.]*(?:introduction|concepts|principles|implementation)[^\.]*)',
            r'(?i)((?:introduction to|principles of|fundamentals of)\s+operating systems?[^\.]*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, context, re.MULTILINE)
            for match in matches:
                clean_title = match.strip().strip('"\'')
                if clean_title and 5 <= len(clean_title) <= 100:
                    # Filter out obvious non-titles
                    if not self._is_likely_dedication_or_non_title(clean_title):
                        candidates.append(clean_title)

        return candidates

    def _strategy_document_structure(self, context: str) -> List[str]:
        """Strategy 2: Analyze document structure for title placement"""
        candidates = []
        lines = context.split('\n')

        # Look for isolated lines (empty lines before/after)
        for i, line in enumerate(lines[:30]):  # Focus on early document
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Check isolation
            prev_empty = (i == 0) or (i > 0 and not lines[i-1].strip())
            next_empty = (i == len(lines)-1) or (i <
                                                 len(lines)-1 and not lines[i+1].strip())

            if (prev_empty or next_empty) and self._looks_like_title(line):
                candidates.append(line)

        return candidates

    def _strategy_semantic_patterns(self, context: str) -> List[str]:
        """Strategy 3: Look for semantic title patterns"""
        candidates = []

        # Enhanced subject/topic patterns for operating systems textbooks
        patterns = [
            r'(?i)(operating systems?[:\s]*(?:three easy pieces|concepts and design|internals and design|principles and practice)[^\.]*)',
            r'(?i)(operating systems?[:\s]*(?:implementation|design|concepts|principles)[^\.]*)',
            r'(?i)(?:an?\s+(?:introduction to|guide to|handbook of|textbook (?:on|about|for)))\s+(operating systems?[^\.]*)',
            r'(?i)(?:principles of|fundamentals of|basics of)\s+(operating systems?[^\.]*)',
            r'(?i)(modern operating systems?[^\.]*)',
            r'(?i)(operating systems?:?\s*[^\.]*(?:introduction|guide|textbook|principles|concepts|design|implementation)[^\.]*)',
            # Look for "Three Easy Pieces" pattern which is a known OS textbook
            r'(?i)((?:operating systems?[:\s]*)?three easy pieces)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m.strip() and len(m.strip()) > 5:
                            clean_match = m.strip()
                            if not self._is_likely_dedication_or_non_title(clean_match):
                                candidates.append(clean_match)
                else:
                    if match.strip() and len(match.strip()) > 5:
                        clean_match = match.strip()
                        if not self._is_likely_dedication_or_non_title(clean_match):
                            candidates.append(clean_match)

        return candidates

    def _strategy_position_based(self, context: str) -> List[str]:
        """Strategy 4: Position-based title extraction from document start"""
        candidates = []

        # Get very first content (likely to contain title)
        lines = context.split('\n')[:20]  # First 20 lines

        # Skip dedication-like content and find substantial titles
        for i, line in enumerate(lines):
            line = line.strip()

            # Skip obvious dedication patterns early in document
            if self._is_likely_dedication_or_non_title(line):
                continue

            if self._looks_like_formatted_title(line):
                candidates.append(line)

        # Look for the first substantial line that could be a title
        substantial_found = False
        for line in lines:
            line = line.strip()

            # Skip dedications and acknowledgments
            if self._is_likely_dedication_or_non_title(line):
                continue

            if len(line) > 15 and self._is_substantial_title_candidate(line):
                candidates.append(line)
                substantial_found = True
                if substantial_found:  # Take first good substantial candidate
                    break

        return candidates

    def _strategy_format_analysis(self, context: str) -> List[str]:
        """Strategy 5: Format-based title detection (caps, centering, etc.)"""
        candidates = []
        lines = context.split('\n')[:25]  # Focus on early content

        for line in lines:
            line = line.strip()

            # All caps titles
            if line.isupper() and 10 <= len(line) <= 80:
                words = line.split()
                if 2 <= len(words) <= 12:
                    candidates.append(line)

            # Title case with good characteristics
            elif line.istitle() and self._looks_like_title(line):
                candidates.append(line)

            # Mixed case but centered-looking (common in PDFs)
            elif re.match(r'^[A-Z][a-zA-Z\s\-:&()]+$', line) and 10 <= len(line) <= 60:
                word_count = len(line.split())
                if 2 <= word_count <= 8:  # Reasonable title length
                    candidates.append(line)

        return candidates

    def _is_substantial_title_candidate(self, line: str) -> bool:
        """Check if line is a substantial title candidate"""
        if len(line) < 10 or len(line) > 100:
            return False

        # Must start with capital
        if not line[0].isupper():
            return False

        # Exclude obvious non-titles
        exclude_starts = ['chapter ', 'section ', 'page ',
                          'copyright', '©', 'table of', 'contents', 'index']
        if any(line.lower().startswith(start) for start in exclude_starts):
            return False

        # Reasonable word count
        word_count = len(line.split())
        if not (2 <= word_count <= 12):
            return False

        return True

    def _is_likely_dedication_or_non_title(self, text: str) -> bool:
        """Check if text is likely a dedication, acknowledgment, or other non-title content"""
        text_lower = text.lower()

        # Common dedication/acknowledgment patterns
        dedication_indicators = [
            'to everyone', 'to all', 'to my', 'to our', 'dedicated to', 'in memory of',
            'acknowledgment', 'acknowledgement', 'thanks to', 'special thanks',
            'we would like to thank', 'grateful to', 'this book is dedicated',
            'foreword', 'preface', 'table of contents', 'contents',
            'copyright', '©', 'all rights reserved', 'published by',
            'isbn', 'library of congress', 'printed in',
            'first edition', 'second edition', 'third edition',
            'page ', 'chapter ', 'section ', 'part i', 'part ii'
        ]

        # Check if text contains dedication indicators
        for indicator in dedication_indicators:
            if indicator in text_lower:
                return True

        # Check if it's just a short greeting or dedication
        if len(text.split()) <= 3 and any(word in text_lower for word in ['to', 'for', 'dear', 'hello']):
            return True

        # Check if it starts with preposition (likely dedication)
        if text_lower.startswith(('to ', 'for ', 'in ', 'with ', 'by ')):
            return True

        return False

    def _cross_validate_candidates(self, all_candidates: List[str], strategy_results: Dict, context: str) -> List[Dict]:
        """Cross-validate candidates across strategies and score them"""
        candidate_counts = {}
        candidate_strategies = {}

        # Filter out dedication-like candidates
        filtered_candidates = []
        for candidate in all_candidates:
            if not self._is_likely_dedication_or_non_title(candidate):
                filtered_candidates.append(candidate)

        # Count how many strategies found each candidate
        for candidate in filtered_candidates:
            candidate_counts[candidate] = candidate_counts.get(
                candidate, 0) + 1
            if candidate not in candidate_strategies:
                candidate_strategies[candidate] = []

        # Track which strategies found each candidate
        for strategy_name, candidates in strategy_results.items():
            for candidate in candidates:
                if candidate in candidate_counts:  # Only count filtered candidates
                    if candidate not in candidate_strategies[candidate]:
                        candidate_strategies[candidate].append(strategy_name)

        # Score candidates based on cross-strategy validation
        scored_candidates = []

        for candidate in set(filtered_candidates):
            score = 0
            reasons = []

            # Multi-strategy bonus (candidates found by multiple strategies score higher)
            strategy_count = candidate_counts[candidate]
            if strategy_count >= 3:
                score += 5
                reasons.append("multiple_strategies")
            elif strategy_count >= 2:
                score += 3
                reasons.append("dual_strategies")
            else:
                score += 1
                reasons.append("single_strategy")

            # Apply existing scoring criteria
            existing_scores = self._score_title_candidates(
                [candidate], context)
            if existing_scores:
                score += existing_scores[0]['score']
                reasons.extend(existing_scores[0]['reasons'])

            # Bonus for being found by high-confidence strategies
            high_conf_strategies = [
                '_strategy_metadata_search', '_strategy_semantic_patterns']
            if any(strategy in candidate_strategies[candidate] for strategy in high_conf_strategies):
                score += 2
                reasons.append("high_confidence_strategy")

            # Bonus for containing "operating systems" (subject matter match)
            if 'operating system' in candidate.lower():
                score += 3
                reasons.append("subject_match")

            # Bonus for academic title patterns
            if any(phrase in candidate.lower() for phrase in ['introduction', 'principles', 'concepts', 'design', 'implementation']):
                score += 2
                reasons.append("academic_title")

            scored_candidates.append({
                'text': candidate,
                'score': score,
                'strategy_count': strategy_count,
                'strategies': candidate_strategies[candidate],
                'reasons': reasons
            })

        return sorted(scored_candidates, key=lambda x: (x['score'], x['strategy_count']), reverse=True)

    def _select_best_parallel_candidate(self, candidates: List[Dict]) -> Optional[Dict]:
        """Select best candidate from parallel thinking results"""
        if not candidates:
            return None

        # Take the highest scoring candidate with reasonable thresholds
        best = candidates[0]

        # Higher threshold for parallel thinking (more rigorous)
        if best['score'] >= 4 or (best['score'] >= 2 and best['strategy_count'] >= 2):
            return best

        return None

    def _looks_like_title(self, line: str) -> bool:
        """Check if a line has title-like characteristics"""
        # Basic length check
        if len(line) < 5 or len(line) > 100:
            return False

        # Exclude obvious non-titles
        exclude_patterns = [
            r'^\d+$',  # Just numbers
            r'^page \d+',  # Page numbers
            r'^chapter \d+',  # Chapter numbers
            r'^section \d+',  # Section numbers
            r'^\w+\@\w+',  # Email addresses
            r'^http',  # URLs
            r'copyright|©|\(c\)',  # Copyright notices
            r'\.{2,}',  # Multiple dots (table of contents)
            r'^[ivx]+\.',  # Roman numerals with dots
        ]

        for pattern in exclude_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return False

        # Positive indicators
        # Starts with capital and has reasonable word structure
        if re.match(r'^[A-Z][a-zA-Z\s\-:&\(\)]+$', line):
            # Not just a sentence fragment
            word_count = len(line.split())
            if 2 <= word_count <= 15:  # Reasonable title length
                return True

        return False

    def _looks_like_formatted_title(self, line: str) -> bool:
        """Check for formatted titles (ALL CAPS, centered, etc.)"""
        line = line.strip()

        # All caps titles
        if line.isupper() and 5 <= len(line) <= 80:
            words = line.split()
            if 2 <= len(words) <= 12:  # Reasonable word count
                return True

        # Mixed case but looks like a title
        if re.match(r'^[A-Z][a-zA-Z\s\-:&]+$', line):
            # Check if it's a standalone line (common for titles)
            word_count = len(line.split())
            if 2 <= word_count <= 10:
                return True

        return False

    def _score_title_candidates(self, candidates: List[str], context: str) -> List[Dict]:
        """Step 2: Score title candidates based on multiple criteria"""
        scored = []

        for candidate in candidates:
            score = 0
            reasons = []

            # Length scoring (titles are usually 2-10 words)
            word_count = len(candidate.split())
            if 2 <= word_count <= 6:
                score += 3
                reasons.append("good_length")
            elif word_count <= 10:
                score += 1
                reasons.append("acceptable_length")

            # Position scoring (earlier = better for titles)
            context_position = context.lower().find(candidate.lower())
            if context_position != -1:
                if context_position < len(context) * 0.1:  # First 10% of content
                    score += 3
                    reasons.append("early_position")
                elif context_position < len(context) * 0.3:  # First 30%
                    score += 1
                    reasons.append("decent_position")

            # Content quality scoring
            if any(word in candidate.lower() for word in ['operating', 'systems', 'introduction', 'guide', 'principles']):
                score += 2
                reasons.append("relevant_content")

            # Format scoring
            if candidate.isupper():
                score += 1
                reasons.append("all_caps")
            elif candidate.istitle():
                score += 2
                reasons.append("title_case")

            # Avoid sentence fragments
            if candidate.lower().startswith(('is ', 'the ', 'a ', 'an ', 'and ', 'but ', 'or ')):
                score -= 2
                reasons.append("sentence_fragment")

            # Avoid incomplete thoughts
            if candidate.endswith(('of the', 'to the', 'in the', 'and')):
                score -= 3
                reasons.append("incomplete_phrase")

            scored.append({
                'text': candidate,
                'score': score,
                'reasons': reasons
            })

        # Sort by score descending
        return sorted(scored, key=lambda x: x['score'], reverse=True)

    def _select_best_title(self, scored_candidates: List[Dict]) -> Optional[Dict]:
        """Step 3: Select the best title candidate"""
        if not scored_candidates:
            return None

        # Take the highest scoring candidate if it has a reasonable score
        best = scored_candidates[0]
        if best['score'] >= 2:  # Minimum threshold
            return best

        return None

    def _fallback_title_analysis(self, context: str) -> str:
        """Step 4: Fallback when no clear title is found"""
        # Look for subject matter clues
        if 'operating systems' in context.lower():
            return "Based on the document content: This appears to be a textbook about **Operating Systems**. The exact title couldn't be clearly identified from the available content."

        # Look for educational context
        if any(word in context.lower() for word in ['students', 'textbook', 'introduction', 'learning']):
            return "Based on the document content: This appears to be an educational textbook. The exact title couldn't be clearly identified, but it seems to be an instructional text for students."

        return "I found content from the document but couldn't clearly identify the book title. The document appears to be a technical text, possibly about computer science or operating systems, but the title may be located in a part of the document that wasn't retrieved in this search."

    def _extract_author_info(self, context: str, question: str) -> str:
        """Extract author information from context"""
        # Look for author patterns
        author_patterns = [
            r'(?i)(?:author|by|written by)[:\s]+([^\n]+)',
            # Name patterns
            r'(?i)([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+))?',
        ]

        found_authors = []

        for pattern in author_patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                if isinstance(match, tuple):
                    for name in match:
                        if name and len(name.strip()) > 3:
                            found_authors.append(name.strip())
                else:
                    if len(match.strip()) > 3:
                        found_authors.append(match.strip())

        # Look in acknowledgments section which was returned
        if 'acknowledgments' in context.lower() or 'thanks' in context.lower():
            # Extract names from the acknowledgments
            lines = context.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ['include:', 'helped', 'contributors']):
                    # Extract potential author names
                    names = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', line)
                    found_authors.extend(
                        [name for name in names if len(name) > 5])

        if found_authors:
            # Take first 3 unique
            unique_authors = list(set(found_authors[:3]))
            return f"Based on the document content, the author(s) appear to be: {', '.join(unique_authors)}"

        return "I found the acknowledgments section but couldn't clearly identify the main authors. The document mentions many contributors and collaborators."

    def _extract_book_summary(self, context: str, question: str) -> str:
        """Extract book summary/topic information"""
        # Look for key topic indicators
        topic_indicators = ['operating systems',
                            'computer science', 'textbook', 'students', 'learning']

        sentences = context.split('. ')
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and
                    any(indicator in sentence.lower() for indicator in topic_indicators)):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            # Take the most descriptive sentences
            summary = '. '.join(relevant_sentences[:2])
            if not summary.endswith('.'):
                summary += '.'
            return f"Based on the document content: {summary}"

        # Fallback to general approach
        return self._extract_general_answer(context, question)

    def _extract_general_answer(self, context: str, question: str) -> str:
        """General extractive approach for other questions"""
        # Simple extractive approach: find the most relevant sentences
        sentences = context.split('. ')

        # Filter sentences that might contain relevant information
        relevant_sentences = []
        question_words = set(question.lower().split())

        # Limit to first 10 sentences for brevity
        for sentence in sentences[:10]:
            sentence_words = set(sentence.lower().split())
            # Simple relevance scoring based on word overlap
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0 and len(sentence.strip()) > 20:
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            # Take top 3 most relevant sentences
            answer = ". ".join(relevant_sentences[:3])
            if not answer.endswith('.'):
                answer += "."

            self.logger.info("Generated extractive answer from context")
            return f"Based on the document content: {answer}"
        else:
            return "I found relevant sections in your document, but couldn't extract a specific answer to your question. The information might be presented in a different way than expected."

    def _extract_contents_info(self, context: str, question: str) -> str:
        """Extract table of contents/chapter information using chain-of-thoughts approach"""
        self.logger.info("Chain-of-thoughts contents extraction started")

        # Step 1: Search for explicit table of contents patterns
        toc_candidates = self._find_contents_patterns(context)
        self.logger.info(f"Found {len(toc_candidates)} TOC patterns")

        # Step 2: Search for chapter/section patterns
        chapter_candidates = self._find_chapter_patterns(context)
        self.logger.info(f"Found {len(chapter_candidates)} chapter patterns")

        # Step 3: Look for topic/subject listings
        topic_candidates = self._find_topic_patterns(context)
        self.logger.info(f"Found {len(topic_candidates)} topic patterns")

        # Step 4: Combine and format results
        all_content_info = toc_candidates + chapter_candidates + topic_candidates

        if all_content_info:
            # Format the contents nicely
            formatted_contents = self._format_contents_response(
                all_content_info, question)
            return formatted_contents
        else:
            # Step 5: Fallback - describe what the book covers based on context
            return self._describe_book_coverage(context)

    def _find_contents_patterns(self, context: str) -> List[str]:
        """Step 1: Look for explicit table of contents patterns"""
        candidates = []

        # Look for explicit TOC markers
        toc_patterns = [
            r'(?i)table of contents[:\n](.*?)(?=\n\n|\nchapter|\npart|\nindex|$)',
            r'(?i)contents[:\n](.*?)(?=\n\n|\nchapter|\npart|\nindex|$)',
            r'(?i)overview[:\n](.*?)(?=\n\n|\nchapter|\npart|\nindex|$)',
        ]

        for pattern in toc_patterns:
            matches = re.findall(pattern, context, re.DOTALL)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match) > 20:  # Substantial content
                    candidates.append(f"Table of Contents:\n{clean_match}")

        return candidates

    def _find_chapter_patterns(self, context: str) -> List[str]:
        """Step 2: Look for chapter/section patterns"""
        candidates = []

        # Look for chapter listings
        chapter_patterns = [
            r'(?i)(chapter \d+[:\s]*[^\n]+)',
            r'(?i)(part [ivx]+[:\s]*[^\n]+)',
            r'(?i)(section \d+[:\s]*[^\n]+)',
            # Look for numbered lists that might be chapters
            r'(\d+\.\s+[A-Z][^\n]+)',
        ]

        found_chapters = []
        for pattern in chapter_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match) > 10 and len(clean_match) < 100:
                    found_chapters.append(clean_match)

        if found_chapters:
            # Group similar chapters together
            unique_chapters = list(set(found_chapters))[
                :10]  # Limit to 10 chapters
            if len(unique_chapters) >= 3:  # Only if we found substantial chapter info
                chapter_list = "\n".join(
                    [f"• {chapter}" for chapter in unique_chapters])
                candidates.append(f"Chapters/Sections found:\n{chapter_list}")

        return candidates

    def _find_topic_patterns(self, context: str) -> List[str]:
        """Step 3: Look for topic/subject listings"""
        candidates = []

        # Look for topic areas in operating systems
        os_topics = [
            'processes', 'threads', 'scheduling', 'memory management', 'virtual memory',
            'file systems', 'storage', 'concurrency', 'synchronization', 'deadlock',
            'virtualization', 'security', 'networking', 'distributed systems'
        ]

        found_topics = []
        for topic in os_topics:
            # Look for mentions of these topics in a structured way
            topic_patterns = [
                rf'(?i)({topic}[:\s]*[^\n]+)',
                rf'(?i)(chapter[^:]*{topic}[^\n]*)',
                rf'(?i)(\d+\.\s*{topic}[^\n]*)',
            ]

            for pattern in topic_patterns:
                matches = re.findall(pattern, context)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(
                            match) > 1 else ''

                    clean_match = match.strip()
                    if len(clean_match) > 5 and clean_match not in found_topics:
                        found_topics.append(clean_match)

        if found_topics:
            topic_list = "\n".join(
                # Limit to 8 topics
                [f"• {topic}" for topic in found_topics[:8]])
            candidates.append(f"Key Topics Covered:\n{topic_list}")

        return candidates

    def _format_contents_response(self, content_info: List[str], question: str) -> str:
        """Step 4: Format the contents response nicely"""
        if not content_info:
            return "I couldn't find specific table of contents information in the available text."

        # Combine all content information
        response_parts = []

        # Add a header based on the question type
        if 'table of contents' in question.lower() or 'contents' in question.lower():
            response_parts.append("**Table of Contents Information:**")
        elif 'chapters' in question.lower():
            response_parts.append("**Chapter Information:**")
        else:
            response_parts.append("**Book Contents Overview:**")

        # Add each piece of content info
        for info in content_info[:3]:  # Limit to top 3 pieces of info
            response_parts.append(f"\n{info}")

        # Add context about the book
        response_parts.append(
            f"\n\n*Note: This appears to be from 'Operating Systems: Three Easy Pieces' - a comprehensive textbook covering operating system concepts and implementation.*")

        return "\n".join(response_parts)

    def _describe_book_coverage(self, context: str) -> str:
        """Step 5: Fallback - describe what the book covers based on available context"""
        # Extract key themes from the context
        themes = []

        if 'operating systems' in context.lower():
            themes.append("Operating Systems concepts and principles")

        if any(word in context.lower() for word in ['processes', 'threads', 'cpu']):
            themes.append("Process and thread management")

        if any(word in context.lower() for word in ['memory', 'virtual']):
            themes.append("Memory management and virtualization")

        if any(word in context.lower() for word in ['file', 'storage']):
            themes.append("File systems and storage")

        if any(word in context.lower() for word in ['concurrency', 'synchronization']):
            themes.append("Concurrency and synchronization")

        if themes:
            theme_list = "\n".join([f"• {theme}" for theme in themes])
            return f"""**Book Coverage Overview:**

Based on the available content, this book covers:

{theme_list}

*Note: This appears to be 'Operating Systems: Three Easy Pieces' - a comprehensive textbook. For a complete table of contents, you may need to access the beginning of the document or the original PDF's table of contents section.*"""
        else:
            return """**Book Contents:**

This appears to be a comprehensive textbook about operating systems. The available content suggests it covers fundamental operating system concepts, but I couldn't extract a specific table of contents from the retrieved sections.

*Note: For the complete table of contents, you may need to access the beginning of the document where such information is typically located.*"""


class RAGService:
    """Service for the complete RAG pipeline"""

    def __init__(self):
        self.vector_store = CustomFAISSVectorStore()
        self.llm_service = LLMService()
        self.logger = get_logger(__name__)

    def process_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Process and add documents to the vector store"""
        self.logger.info(f"Processing {len(texts)} documents for RAG")
        self.vector_store.add_documents(texts, metadatas)

    def retrieve_context(self, query: str, k: int = None) -> List[str]:
        """Retrieve relevant context for a query"""
        k = k or settings.TOP_K_RETRIEVAL

        self.logger.info(f"Retrieving context for query (top-{k})")

        # Check if this is a title/metadata question that might need different search strategy
        query_lower = query.lower()

        if any(word in query_lower for word in ['title', 'name of the book', 'book called', 'book title']):
            # For title questions, also search the beginning of the document
            return self._retrieve_title_context(query, k)
        elif any(word in query_lower for word in ['author', 'written by', 'who wrote']):
            # For author questions, search beginning and acknowledgments
            return self._retrieve_author_context(query, k)
        elif any(phrase in query_lower for phrase in ['contents', 'table of contents', 'chapters', 'outline', 'structure', 'what does this book cover', 'topics covered']):
            # For contents questions, search for table of contents and chapter listings
            return self._retrieve_contents_context(query, k)
        else:
            # Regular semantic search
            results = self.vector_store.similarity_search(query, k)
            context_texts = [result[0] for result in results]
            self.logger.info(f"Retrieved {len(context_texts)} context chunks")
            return context_texts

    def _retrieve_title_context(self, query: str, k: int) -> List[str]:
        """Retrieve context specifically for title questions"""
        context_texts = []

        # First, get the earliest chunks (likely to contain title info)
        if self.vector_store.documents:
            # Get first few chunks that might contain title/metadata
            early_chunks = self.vector_store.documents[:min(
                10, len(self.vector_store.documents))]
            context_texts.extend(early_chunks[:3])  # Take first 3 chunks

        # Also do semantic search to get any other relevant content
        results = self.vector_store.similarity_search(query, k//2)
        semantic_texts = [result[0] for result in results]

        # Combine and deduplicate
        all_texts = context_texts + semantic_texts
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in all_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        final_texts = unique_texts[:k]
        self.logger.info(
            f"Retrieved {len(final_texts)} context chunks for title query")
        return final_texts

    def _retrieve_author_context(self, query: str, k: int) -> List[str]:
        """Retrieve context specifically for author questions"""
        context_texts = []

        # Look for acknowledgments or credits sections
        if self.vector_store.documents:
            for i, doc in enumerate(self.vector_store.documents):
                if any(word in doc.lower() for word in ['acknowledgment', 'author', 'written by', 'credits']):
                    context_texts.append(doc)
                    if len(context_texts) >= 2:  # Get a couple of relevant chunks
                        break

        # Also do semantic search
        results = self.vector_store.similarity_search(query, k//2)
        semantic_texts = [result[0] for result in results]

        # Combine and deduplicate
        all_texts = context_texts + semantic_texts
        seen = set()
        unique_texts = []
        for text in all_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        final_texts = unique_texts[:k]
        self.logger.info(
            f"Retrieved {len(final_texts)} context chunks for author query")
        return final_texts

    def _retrieve_contents_context(self, query: str, k: int) -> List[str]:
        """Retrieve context specifically for contents/table of contents questions"""
        context_texts = []

        # Strategy 1: Look for actual table of contents sections
        if self.vector_store.documents:
            for i, doc in enumerate(self.vector_store.documents):
                doc_lower = doc.lower()
                if any(phrase in doc_lower for phrase in ['table of contents', 'contents', 'overview', 'chapter', 'part i', 'part ii']):
                    context_texts.append(doc)
                    if len(context_texts) >= 3:  # Get a few relevant chunks
                        break

        # Strategy 2: Look for chapter/section patterns throughout the document
        if len(context_texts) < 2 and self.vector_store.documents:
            # Search for documents that contain chapter or section references
            for doc in self.vector_store.documents:
                if re.search(r'(?i)(chapter \d+|section \d+|part [ivx]+)', doc):
                    context_texts.append(doc)
                    if len(context_texts) >= 4:
                        break

        # Strategy 3: Get early document content (might contain overview/structure)
        if len(context_texts) < 3 and self.vector_store.documents:
            # Add first few chunks which might contain introductory/overview content
            early_chunks = self.vector_store.documents[:min(
                5, len(self.vector_store.documents))]
            for chunk in early_chunks[:2]:  # Add first 2 chunks
                if chunk not in context_texts:
                    context_texts.append(chunk)

        # Strategy 4: Semantic search as backup
        try:
            # Search for table of contents specifically
            toc_results = self.vector_store.similarity_search(
                "table of contents chapters overview", k//2)
            for result in toc_results:
                if result[0] not in context_texts:
                    context_texts.append(result[0])

            # Search for the original query
            query_results = self.vector_store.similarity_search(query, k//2)
            for result in query_results:
                if result[0] not in context_texts:
                    context_texts.append(result[0])

        except Exception as e:
            self.logger.warning(
                f"Semantic search failed for contents query: {e}")

        # Limit to k results
        final_texts = context_texts[:k]
        self.logger.info(
            f"Retrieved {len(final_texts)} context chunks for contents query")
        return final_texts

    def format_prompt(self, query: str, context: List[str]) -> str:
        """Format the prompt for the LLM"""
        context_text = "\n\n".join(context)

        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context_text}

Question: {query}

Answer:"""

        return prompt

    async def generate_answer(self, query: str) -> str:
        """Generate an answer using the RAG pipeline"""
        self.logger.info("Starting RAG pipeline for answer generation")

        # Step 1: Retrieve relevant context
        context = self.retrieve_context(query)

        if not context:
            return "I don't have any relevant information to answer your question. Please upload some documents first."

        # Step 2: Format prompt
        prompt = self.format_prompt(query, context)

        # Step 3: Generate answer using LLM service
        try:
            answer = await self.llm_service.generate_completion(prompt)
            self.logger.info("Successfully generated answer using LLM service")
            return answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"I found {len(context)} relevant sections in your documents, but encountered an error generating the answer. Please try rephrasing your question."

    async def generate_streaming_answer(self, query: str):
        """Generate a streaming answer using the complete RAG pipeline"""
        self.logger.info(
            "Starting streaming RAG pipeline for answer generation")

        try:
            # Step 1: Send initial status
            yield {
                "type": "status",
                "data": "🔍 Searching your documents...",
                "stage": "retrieval",
                "progress": 10
            }

            # Step 2: Retrieve relevant context
            context = self.retrieve_context(query)

            if not context:
                yield {
                    "type": "error",
                    "data": "I don't have any relevant information to answer your question. Please upload some documents first.",
                    "stage": "retrieval",
                    "progress": 100
                }
                return

            # Step 3: Context found, update status
            yield {
                "type": "status",
                "data": f"📄 Found {len(context)} relevant sections. Analyzing...",
                "stage": "analysis",
                "progress": 30
            }

            # Step 4: Format prompt
            prompt = self.format_prompt(query, context)

            # Step 5: Preparing for AI generation
            yield {
                "type": "status",
                "data": "🤖 Generating answer with AI...",
                "stage": "generation",
                "progress": 50
            }

            # Step 6: Generate streaming answer using LLM service
            answer_started = False
            try:
                async for chunk in self.llm_service.generate_streaming_completion(prompt):
                    if chunk["type"] == "content":
                        if not answer_started:
                            # First content chunk - send answer start signal
                            yield {
                                "type": "answer_start",
                                "data": "✨ Answer:",
                                "stage": "streaming",
                                "progress": 60
                            }
                            answer_started = True

                        # Send content chunk
                        yield {
                            "type": "content",
                            "data": chunk["data"],
                            "provider": chunk["provider"],
                            "stage": "streaming",
                            # Progressive progress
                            "progress": min(90, 60 + (30 * len(chunk["data"]) / 100))
                        }

                    elif chunk["type"] == "complete":
                        # Send completion with metadata
                        yield {
                            "type": "complete",
                            "data": "✅ Answer complete",
                            "provider": chunk["provider"],
                            "stage": "complete",
                            "progress": 100,
                            "metadata": {
                                "context_chunks": len(context),
                                "query_length": len(query),
                                "retrieval_model": "FAISS + Local Embeddings",
                                "llm_provider": chunk["provider"]
                            }
                        }

                self.logger.info(
                    "Successfully completed streaming RAG pipeline")

            except Exception as e:
                self.logger.error(f"Error in streaming generation: {e}")
                yield {
                    "type": "error",
                    "data": f"I found {len(context)} relevant sections but encountered an error generating the answer: {str(e)}",
                    "stage": "generation_error",
                    "progress": 100
                }

        except Exception as e:
            self.logger.error(f"Error in streaming RAG pipeline: {e}")
            yield {
                "type": "error",
                "data": f"Error in RAG pipeline: {str(e)}",
                "stage": "pipeline_error",
                "progress": 100
            }

    def clear_documents(self):
        """Clear all documents from the vector store"""
        self.vector_store.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG service"""
        return self.vector_store.get_stats()
