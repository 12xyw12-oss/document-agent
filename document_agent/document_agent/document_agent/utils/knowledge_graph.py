import json
import os
from config import KNOWLEDGE_GRAPH_PATH

class SimpleKnowledgeGraph:
    def __init__(self):
        self.graph_path = os.path.join(KNOWLEDGE_GRAPH_PATH, "graph.json")
        self.entities = {}
        self.relations = []
        self._load()
    
    def _load(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entities = data.get("entities", {})
                self.relations = data.get("relations", [])
    
    def _save(self):
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump({
                "entities": self.entities,
                "relations": self.relations
            }, f, ensure_ascii=False, indent=2)
    
    def add_entity(self, entity, attributes=None):
        if entity not in self.entities:
            self.entities[entity] = attributes or {}
        else:
            self.entities[entity].update(attributes or {})
        self._save()
    
    def add_relation(self, entity1, relation, entity2, source=None):
        relation_tuple = (entity1, relation, entity2, source)
        if relation_tuple not in self.relations:
            self.relations.append(relation_tuple)
            self._save()
    
    def get_related_entities(self, entity, relation=None):
        results = []
        for e1, rel, e2, src in self.relations:
            if e1 == entity:
                if relation is None or rel == relation:
                    results.append((e2, rel, src))
            elif e2 == entity:
                if relation is None or rel == relation:
                    results.append((e1, rel, src))
        return results
    
    def search_entities(self, keyword):
        return [entity for entity in self.entities if keyword.lower() in entity.lower()]
    
    def get_entity_attributes(self, entity):
        return self.entities.get(entity, {})