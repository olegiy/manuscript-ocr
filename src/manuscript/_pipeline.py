import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .data import Page
from .detectors import EAST
from .recognizers import TRBA
from .utils import read_image, visualize_page


class Pipeline:
    """
    High-level OCR pipeline combining text detection and recognition.

    The Pipeline class orchestrates EAST detector and TRBA recognizer to perform
    complete OCR workflow: detection → crop extraction → recognition → result merging.

    Attributes
    ----------
    detector : EAST
        Text detector instance
    recognizer : TRBA
        Text recognizer instance
    min_text_size : int
        Minimum text box size in pixels (width and height)

    Examples
    --------
    Create pipeline with default models:

    >>> from manuscript import Pipeline
    >>> pipeline = Pipeline()
    >>> result = pipeline.predict("document.jpg")
    >>> text = pipeline.get_text(result["page"])
    >>> print(text)

    Create pipeline with custom models:

    >>> from manuscript import Pipeline
    >>> from manuscript.detectors import EAST
    >>> from manuscript.recognizers import TRBA
    >>> detector = EAST(weights="east_50_g1", score_thresh=0.8)
    >>> recognizer = TRBA(weights="trba_lite_g1", device="cuda")
    >>> pipeline = Pipeline(detector=detector, recognizer=recognizer)
    """

    def __init__(
        self,
        detector: Optional[EAST] = None,
        recognizer: Optional[TRBA] = None,
        min_text_size: int = 5,
    ):
        """
        Initialize OCR pipeline.

        Parameters
        ----------
        detector : EAST, optional
            Text detector instance. If None, creates default EAST detector.
        recognizer : TRBA, optional
            Text recognizer instance. If None, creates default TRBA recognizer.
        min_text_size : int, optional
            Minimum text size in pixels. Boxes smaller than this will be
            filtered out before recognition. Default is 5.
        """
        self.detector = detector if detector is not None else EAST()
        self.recognizer = recognizer if recognizer is not None else TRBA()
        self.min_text_size = min_text_size

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ) -> Union[Dict, tuple]:
        """
        Run OCR pipeline on a single image.

        Parameters
        ----------
        image : str, Path, numpy.ndarray, or PIL.Image
            Input image. Can be:
            - Path to image file (str or Path)
            - RGB numpy array with shape (H, W, 3) in uint8
            - PIL Image object
        recognize_text : bool, optional
            If True, performs both detection and recognition.
            If False, performs only detection. Default is True.
        vis : bool, optional
            If True, returns visualization image along with results.
            Default is False.
        profile : bool, optional
            If True, prints timing information for each pipeline stage.
            Default is False.

        Returns
        -------
        dict or tuple
            If vis=False:
                dict with keys:
                - "page" : Page object with detection/recognition results

            If vis=True:
                tuple of (result_dict, vis_image)

        Examples
        --------
        Basic usage:

        >>> pipeline = Pipeline()
        >>> result = pipeline.predict("document.jpg")
        >>> page = result["page"]
        >>> print(page.blocks[0].lines[0].words[0].text)

        Detection only:

        >>> result = pipeline.predict("document.jpg", recognize_text=False)
        >>> # Words will have polygon and detection_confidence but no text

        With visualization:

        >>> result, vis_img = pipeline.predict("document.jpg", vis=True)
        >>> vis_img.show()

        With profiling:

        >>> result = pipeline.predict("document.jpg", profile=True)
        # Prints timing for each stage
        """
        start_time = time.time()

        # ---- DETECTION ----
        t0 = time.time()
        detection_result = self.detector.predict(
            image, return_maps=False, sort_reading_order=True
        )
        page: Page = detection_result["page"]

        if profile:
            print(f"Detection: {time.time() - t0:.3f}s")

        # ---- If recognition not needed ----
        if not recognize_text:
            result = {"page": page}

            if vis:
                img_array = read_image(image)
                pil_img = (
                    image
                    if isinstance(image, Image.Image)
                    else Image.fromarray(img_array)
                )
                vis_img = visualize_page(pil_img, page, show_order=False)
                return result, vis_img

            return result

        # ---- LOAD IMAGE FOR CROPPING ----
        t0 = time.time()
        image_array = read_image(image)
        if profile:
            print(f"Load image for crops: {time.time() - t0:.3f}s")

        # ---- EXTRACT WORD CROPS ----
        t0 = time.time()
        word_images = []
        word_objects = []  # Keep references to Word objects for updating

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    poly = np.array(word.polygon, dtype=np.int32)
                    x_min, y_min = np.min(poly, axis=0)
                    x_max, y_max = np.max(poly, axis=0)

                    width = x_max - x_min
                    height = y_max - y_min

                    # Filter by minimum size
                    if width >= self.min_text_size and height >= self.min_text_size:
                        region_image = self._extract_word_image(image_array, poly)
                        if region_image is not None and region_image.size > 0:
                            word_images.append(region_image)
                            word_objects.append(word)

        if profile:
            print(f"Extract {len(word_images)} crops: {time.time() - t0:.3f}s")

        # ---- RECOGNITION ----
        if word_images:
            t0 = time.time()
            recognition_results = self.recognizer.predict(word_images, batch_size=32)
            if profile:
                print(f"Recognition: {time.time() - t0:.3f}s")

            # Update Word objects with recognition results
            for word_obj, result in zip(word_objects, recognition_results):
                word_obj.text = result["text"]
                word_obj.recognition_confidence = result["confidence"]

        if profile:
            print(f"Pipeline total: {time.time() - start_time:.3f}s")

        result = {"page": page}

        if vis:
            pil_img = (
                image
                if isinstance(image, Image.Image)
                else Image.fromarray(image_array)
            )
            vis_img = visualize_page(pil_img, page, show_order=True)
            return result, vis_img

        return result

    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ) -> List[Union[Dict, tuple]]:
        """
        Process multiple images in sequence.

        Parameters
        ----------
        images : list
            List of images to process. Each can be str, Path, numpy.ndarray, or PIL.Image.
        recognize_text : bool, optional
            If True, performs recognition. Default is True.
        vis : bool, optional
            If True, returns visualization for each image. Default is False.
        profile : bool, optional
            If True, prints timing information. Default is False.

        Returns
        -------
        list
            List of results (dict or tuple) for each input image.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> images = ["page1.jpg", "page2.jpg", "page3.jpg"]
        >>> results = pipeline.process_batch(images)
        >>> for result in results:
        ...     text = pipeline.get_text(result["page"])
        ...     print(text)
        """
        results = []
        for img in images:
            res = self.predict(
                img, recognize_text=recognize_text, vis=vis, profile=profile
            )
            results.append(res)
        return results

    def get_text(self, page: Page) -> str:
        """
        Extract plain text from Page object.

        Parameters
        ----------
        page : Page
            Page object with recognition results.

        Returns
        -------
        str
            Extracted text with lines separated by newlines.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> result = pipeline.predict("document.jpg")
        >>> text = pipeline.get_text(result["page"])
        >>> print(text)
        """
        lines = []
        for block in page.blocks:
            for line in block.lines:
                # Extract text from words in the line
                texts = [w.text for w in line.words if w.text]
                if texts:
                    lines.append(" ".join(texts))
        return "\n".join(lines)

    def correct_text_with_llm(
        self,
        text: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Correct OCR errors in plain text using LLM.

        Parameters
        ----------
        text : str
            OCR text to correct.
        api_key : str, optional
            OpenAI API key. If None, uses "dummy_key" (for local models).
        api_url : str, optional
            API endpoint URL. If None, uses OpenAI default.
        model : str, optional
            Model name. Default is "gpt-4o-2024-08-06".
        temperature : float, optional
            Sampling temperature. Default is 0.0.
        system_prompt : str, optional
            Custom system prompt. If None, uses default correction prompt.

        Returns
        -------
        str
            Corrected text.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> result = pipeline.predict("document.jpg")
        >>> text = pipeline.get_text(result["page"])
        >>> corrected = pipeline.correct_text_with_llm(text, api_url="https://demo.ai.sfu-kras.ru/v1")
        """
        import openai

        # Configure OpenAI client
        openai.api_key = api_key or "dummy_key"
        if api_url:
            openai.api_base = api_url
            openai.api_type = "openai"
            openai.api_version = None

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are an advanced OCR text correction assistant.

        Your task is to restore the original intended human-written text from noisy OCR output.

        RULES:
        1. Correct all OCR-related errors:
        - misrecognized characters (e.g., "0"→"О", "Cl"→"Л", "rn"→"m")
        - broken or merged words ("чувств а" → "чувства", "мир а" → "мира")
        - repeated fragments or duplicated words caused by OCR
        - missing or extra letters
        - incorrect casing and random capitalization
        - incorrect endings, cases, verb forms, grammatical errors caused by OCR noise

        2. Improve readability by restoring proper:
        - spelling
        - grammar
        - word forms
        - punctuation
        - sentence boundaries
        - paragraph structure

        3. Preserve:
        - original line breaks
        - original paragraph boundaries
        - the overall semantic meaning
        - the style of the author (do NOT rewrite or paraphrase)

        4. Do NOT:
        - add new content
        - change the meaning
        - summarize or rewrite stylistically
        - remove lines
        - merge paragraphs
        - introduce new ideas or interpretations

        5. Your output must contain ONLY the corrected text.
        No explanations, no comments, no metadata.

        Your goal is to produce the cleanest, most accurate reconstruction of the original text while preserving structure."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Correct OCR errors in this text:\n\n{text}"},
        ]

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        corrected_text = response.choices[0].message.content.strip()
        return corrected_text

    def _extract_word_image(
        self, image: np.ndarray, polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)

            h, w = image.shape[:2]
            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(w, int(x_max))
            y2 = min(h, int(y_max))

            region_image = image[y1:y2, x1:x2]

            return region_image if region_image.size > 0 else None
        except Exception:
            return None

    def correct_with_llm(
        self,
        page: Page,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> Page:
        """Correct OCR errors using LLM via OpenAI library with 1-to-1 word mapping."""
        import json
        import openai

        # Configure OpenAI client
        openai.api_key = api_key or "dummy_key"
        if api_url:
            openai.api_base = api_url
            openai.api_type = "openai"
            openai.api_version = None

        # Extract only text structure (without coordinates and confidence)
        structure = []
        for block in page.blocks:
            block_lines = []
            for line in block.lines:
                words_text = [w.text if w.text else "" for w in line.words]
                block_lines.append(words_text)
            structure.append(block_lines)

        structure_json = json.dumps(structure, ensure_ascii=False, indent=2)

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are an advanced OCR text correction assistant.

        Your task is to restore the original intended human-written text from noisy OCR output.

        IMPORTANT: You MUST return a JSON array with EXACTLY the same structure as the input:
        - Same number of blocks (outer arrays)
        - Same number of lines in each block (middle arrays)
        - Same number of words in each line (inner strings)
        This is a strict 1-to-1 mapping. Do NOT add, remove, merge, or split any elements.

        RULES:
        1. Correct all OCR-related errors:
        - misrecognized characters (e.g., "0"→"О", "Cl"→"Л", "rn"→"m")
        - broken or merged words ("чувств а" → "чувства", "мир а" → "мира")
        - repeated fragments or duplicated words caused by OCR
        - missing or extra letters
        - incorrect casing and random capitalization
        - incorrect endings, cases, verb forms, grammatical errors caused by OCR noise
        - hyphenated words split across lines due to line breaks (e.g., "приме-", "ром" → "приме-", "ром" BUT corrected for OCR errors)

        2. Improve readability by restoring proper:
        - spelling within each word (even if it's part of a hyphenated word)
        - grammar
        - word forms
        - punctuation
        - DO NOT merge words split by line breaks - keep them separate

        3. Preserve:
        - the EXACT structure: same number of blocks, lines, and words
        - words split by hyphens across lines must stay split in separate array elements
        - original line breaks
        - the overall semantic meaning
        - the style of the author (do NOT rewrite or paraphrase)

        4. Do NOT:
        - add new content
        - change the meaning
        - summarize or rewrite stylistically
        - merge or split words (keep 1-to-1 mapping)
        - remove or add words
        - merge hyphenated words across line breaks
        - wrap the array in an object - return the array directly

        5. Your output must be ONLY a JSON array matching the input structure EXACTLY.
        No explanations, no comments, no metadata, no wrapping object.

        Example:
        Input: [[[word1, word2], [word3]], [[word4]]]
        Output: [[[corrected1, corrected2], [corrected3]], [[corrected4]]]

        Each word in the input array corresponds to exactly one word in the output array.
        Hyphenated words at line breaks stay as separate words but with corrected spelling."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Correct OCR errors in words, keeping the EXACT same structure:\n\n{structure_json}",
            },
        ]

        # Call OpenAI API with response_format
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        corrected_text = response.choices[0].message.content.strip()

        # Clean up markdown if present
        if corrected_text.startswith("```json"):
            corrected_text = corrected_text[7:]
        if corrected_text.startswith("```"):
            corrected_text = corrected_text[3:]
        if corrected_text.endswith("```"):
            corrected_text = corrected_text[:-3]
        corrected_text = corrected_text.strip()

        # Parse JSON - handle both direct array and wrapped object
        parsed = json.loads(corrected_text)
        if isinstance(parsed, dict):
            # If model wrapped it in object, try to extract array from various possible keys
            for key in ["corrected", "words", "result", "structure", "blocks", "data"]:
                if key in parsed and isinstance(parsed[key], list):
                    corrected_structure = parsed[key]
                    break
            else:
                # If no known key found, raise error with helpful message
                raise ValueError(
                    f"LLM returned object with unexpected keys: {list(parsed.keys())}. "
                    f"Expected array or object with 'corrected'/'words'/'result' key."
                )
        else:
            corrected_structure = parsed

        # Map corrected words back to page (1-to-1 mapping)
        for block_idx, block in enumerate(page.blocks):
            if block_idx >= len(corrected_structure):
                break
            corrected_block = corrected_structure[block_idx]
            if not isinstance(corrected_block, list):
                continue
            for line_idx, line in enumerate(block.lines):
                if line_idx >= len(corrected_block):
                    break
                corrected_words = corrected_block[line_idx]
                if not isinstance(corrected_words, list):
                    continue
                for word_idx, word in enumerate(line.words):
                    if word_idx < len(corrected_words):
                        word.text = corrected_words[word_idx]

        return page
