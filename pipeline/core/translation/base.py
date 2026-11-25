from pipeline.core.translation.deepseek import DeepseekApi
from pipeline.core.translation.gemini import GeminiApi
from pipeline.core.translation.groq import GroqApi


def _text_objects(node) -> list[dict]:
    """Every {jp_text, tr_text} object in a reply, in order, at any depth."""
    if isinstance(node, dict):
        if "tr_text" in node:
            return [node]
        return [obj for value in node.values() for obj in _text_objects(value)]
    if isinstance(node, list):
        return [obj for value in node for obj in _text_objects(value)]
    return []


def _translation(node) -> str:
    text = node.get("tr_text") if isinstance(node, dict) else None
    return text.strip() if isinstance(text, str) else ""


def fit_to_request(request: dict, reply: dict) -> tuple[dict, int]:
    """The request ({page: {bubble: {cluster: {jp_text, tr_text}}}}) with tr_text filled
    from the reply, and how many clusters got a translation.

    Models don't always keep the structure: Groq sometimes drops the page level and
    returns the bubbles at the top. Each cluster is looked up by position first, then
    by its jp_text among all translated objects in the reply, then (for a reply with
    exactly one object per cluster) by order. Keys are strings, as in a JSON reply.
    """
    objects = _text_objects(reply)
    by_text: dict[str, list[str]] = {}
    for obj in objects:
        if isinstance(obj.get("jp_text"), str) and _translation(obj):
            by_text.setdefault(obj["jp_text"].strip(), []).append(_translation(obj))
    single_page = len(request) == 1
    clusters_total = sum(len(clusters) for bubbles in request.values() for clusters in bubbles.values())
    same_count = len(objects) == clusters_total

    result, filled, order = {}, 0, 0
    for page, bubbles in request.items():
        page_key = str(page)
        page_reply = reply.get(page_key) if isinstance(reply, dict) else None
        result[page_key] = {}
        for bubble, clusters in bubbles.items():
            result[page_key][str(bubble)] = {}
            for cluster, item in clusters.items():
                candidates = [page_reply, reply if single_page else None]  # with / without the page level
                text = ""
                for root in candidates:
                    node = root.get(str(bubble)) if isinstance(root, dict) else None
                    node = node.get(str(cluster)) if isinstance(node, dict) else None
                    if _translation(node) and (not isinstance(node.get("jp_text"), str)
                                               or node["jp_text"].strip() == (item.get("jp_text") or "").strip()):
                        text = _translation(node)
                        break
                if not text:
                    matches = by_text.get((item.get("jp_text") or "").strip())
                    if matches:
                        text = matches.pop(0)
                if not text and same_count:
                    text = _translation(objects[order])
                order += 1
                filled += bool(text)
                result[page_key][str(bubble)][str(cluster)] = {"jp_text": item.get("jp_text"), "tr_text": text}
    return result, filled



class TranslationBase:
    def __init__(self, api_config: dict, translation_config: dict):
        self.api_config = api_config
        self.primary_api = api_config.get("selected")
        
        # Initialize both APIs if keys are available
        self.apis = {}
        
        # Setup Gemini API
        gemini_config = api_config.get("gemini", {})
        gemini_key = gemini_config.get("api_key")
        if gemini_key:
            self.apis["gemini"] = GeminiApi(gemini_key, gemini_config.get("model"))
            
        # Setup DeepSeek API  
        deepseek_config = api_config.get("deepseek", {})
        deepseek_key = deepseek_config.get("api_key")
        if deepseek_key:
            self.apis["deepseek"] = DeepseekApi(deepseek_key, deepseek_config.get("model"))
        
        # Setup Groq API
        groq_config = api_config.get("groq", {})
        groq_key = groq_config.get("api_key")
        if groq_key:
            self.apis["groq"] = GroqApi(groq_key, groq_config.get("model"))
        
        print(f"Available APIs: {list(self.apis.keys())}")
        print(f"Primary API: {self.primary_api}")


    def batch_translation(self, batch_dict: dict) -> dict:
        if not self.apis:
            print("No API keys provided for translation")
            return batch_dict
            
        # Try primary API first
        if self.primary_api in self.apis:
            print(f"Attempting translation with primary API: {self.primary_api}")
            result = self._try_translation(self.primary_api, batch_dict)
            if result:
                return result
                
        # Try fallback APIs
        for api_name, api_instance in self.apis.items():
            if api_name != self.primary_api:
                print(f"Primary API failed, trying fallback: {api_name}")
                result = self._try_translation(api_name, batch_dict)
                if result:
                    return result
                    
        print("All APIs failed, returning original batch")
        return batch_dict
        
    def _try_translation(self, api_name: str, batch_dict: dict) -> dict:
        try:
            api_instance = self.apis[api_name]
            result = api_instance.translate(batch_dict)
            
            # Put the reply back into the request's shape: the reply's own structure
            # can differ (see fit_to_request), and typesetting looks clusters up by it.
            fitted, filled = fit_to_request(batch_dict, result or {})
            total = sum(len(clusters) for bubbles in batch_dict.values() for clusters in bubbles.values())
            if filled:
                print(f"✓ {api_name} API succeeded: {filled}/{total} texts translated")
                return fitted
            print(f"✗ {api_name} API returned no usable translations")
            return None
                
        except Exception as e:
            print(f"✗ {api_name} API failed with exception: {str(e)}")
            return None
        