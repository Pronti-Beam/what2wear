from application.bootstrap.constants import (
    JACKET_CATEGORY_NAME, ONE_PIECE_CATEGORY_NAME, TOP_CATEGORY_NAME,
    BOTTOM_CATEGORY_NAME, SHOE_CATEGORY_NAME, ACCESSORY_CATEGORY_NAME,
    CATEGORIES, WEATHER_MAPPINGS,
    LSTMDirection)

from application.bootstrap.constants import EMBEDDING_SERVICE_API_VERSION
from application.bootstrap.constants import  OUTFIT_ID, OUTFIT_DATA
from application.ml.what2wear.inference import load_bilstm_vse, predict_single_direction, run_lstm
from application.services.EmbeddingService import EmbeddingService
from application.util.exceptions import NotEnoughClosetItemsException, OutfitRetrievalException, OutfitGenerationError
from collections import Counter
from flask import current_app as app
import json
import numpy as np
from torch.nn import Module
import random
import torch

PRE_GENERATED_OUTFITS_FILE_PATH = "./data/pre_generated_outfits.json"
NUM_ITEMS_REMOVED_FROM_CANDIDATES = 1
COMPATIBILITY_SCORE_THRESHOLD = 0
OUTFITS_START_END_IDS_FILE = './data/outfits_start_end_ids.json'


class Item:
    def __init__(self, item_id: int, occasion_tags: list,
                 embedding: torch.Tensor, category: str, image_url:str, type_name: str):
        self.item_id = item_id
        self.occasion_tags = occasion_tags
        self.embedding = embedding
        self.category = category
        self.type_name = type_name
        self.image_url = image_url


class OutfitItemCandidates:
    def __init__(self, items: list):
        self.items = items
        self.categories_count = Counter([item.category for item in self.items])

    def compute_normalized_embeddings(self):
        embeddings = torch.stack([item.embedding for item in self.items])
        normalized_embeddings = torch.nn.functional.normalize(embeddings,
                                                              p=2,
                                                              dim=1)
        return normalized_embeddings

    def update_items(self, outfit: dict, query_item: Item) -> None:
        items = self._find_item_to_remove(outfit, query_item)
        for item in items:
            app.logger.info(
                f'Removing item {item.item_id} with category {item.category}')
            self.items.remove(item)
            self.categories_count[item.category] -= 1

    def _find_item_to_remove(self, outfit, query_item):
        candidate_categories = []
        for category, item in outfit.items():
            if self.categories_count[
                    category] > 1 and category != query_item.category:
                candidate_categories.append(category)
        if len(candidate_categories) < NUM_ITEMS_REMOVED_FROM_CANDIDATES:
            raise NotEnoughClosetItemsException
        categories = random.sample(candidate_categories,
                                   k=NUM_ITEMS_REMOVED_FROM_CANDIDATES)
        items_to_remove = []
        for category in categories:
            items_to_remove.append(outfit[category])
        return items_to_remove

    def get_item_by_index(self, idx):
        return self.items[idx]

    def get_items_by_category(self, category):
        return [item for item in self.items if item.category == category]

    def contains_sufficent_items(self):
        if self.categories_count[ONE_PIECE_CATEGORY_NAME] == 0 and (
                self.categories_count[TOP_CATEGORY_NAME] == 0
                or self.categories_count[BOTTOM_CATEGORY_NAME] == 0):
            app.logger.info(
                f"{self.categories_count[ONE_PIECE_CATEGORY_NAME]}")
            app.logger.info(f"{self.categories_count[TOP_CATEGORY_NAME]}")
            app.logger.info(f"{self.categories_count[BOTTOM_CATEGORY_NAME]}")

            return False
        return True


class Closet:
    def __init__(self, items: list):
        self.items = items
        self.item_count_by_category = Counter(
            [item.category for item in self.items])

    def get_items_by_occasion(self, occasion: str) -> list:
        return [item for item in self.items if occasion in item.occasion_tags]

    def get_items_by_category(self, category: str) -> list:
        return [item for item in self.items if item.category == category]

    def get_items_by_type_nem(self, type_name: str) -> list:
        return [item for item in self.items if item.type_name == type_name]

    def get_items_by_id(self, item_id: int) -> Item:
        for item in self.items:
            if item.item_id == item_id:
                return item
        error_message = f"Query item id {item_id} not in closet"
        raise OutfitGenerationError(error_message)

    def generate_query_items(self):
        query_items = []
        for category in CATEGORIES:
            if self.item_count_by_category[category] < 2:
                continue
            query_item = random.sample(
                [item for item in self.items if item.category == category],
                k=1)[0]
            query_items.append(query_item)
        return query_items

class OutfitRecommendationService:
    def __init__(self):
        self.embedding_service = EmbeddingService(
            EMBEDDING_SERVICE_API_VERSION)
        self.model = load_bilstm_vse()
        
    def recommend_outfits_for_item(self, user_id: int, item_id: int,
                                   item_occasions: list, closet: list):
        item_embeddings = self._get_closet_item_embeddings(user_id)
        closet_items = create_closet_items(closet, item_embeddings)
        closet = Closet(closet_items)
        item = closet.get_items_by_id(item_id)
        outfits = {}
        for occasion in item_occasions:
            candidate_items = OutfitItemCandidates(
                closet.get_items_by_occasion(occasion))
            app.logger.info(
                f"Candidate items length: {len(candidate_items.items)}")
            if not candidate_items.contains_sufficent_items():
                warning_msg = f"Cannot generate outfits for {occasion}, insufficient items"
                app.logger.warning(warning_msg)
                continue
            app.logger.info(
                f'Generating set of outfits for occasion {occasion}')
            try:
                outfit_set = generate_outfit_set_for_item(
                    item, candidate_items, self.model)
                outfits[occasion] = serialize_outfits(outfit_set)
            except OutfitGenerationError as e:
                app.logger.warning(
                    f'Error generating outfit for occasion {occasion}, {e}')
                continue
        return outfits

    def retreive_pregenerated_outfits(self, gender, occasion,
                                      stock_image_item_id_map):
        outfits = load_outfits_for_user(gender, occasion)
        mapped_outfits = []
        for outfit_data in outfits.values():
            mapped_outfit = map_basic_ids_to_item_ids(outfit_data['outfit'],
                                                      stock_image_item_id_map)
            outfit_data['outfit'] = mapped_outfit
            mapped_outfits.append(outfit_data)
        return mapped_outfits

    def recommend_outfits_using_closet(self, occasions: list, closet: dict,
                                       user_id: int) -> dict:
        item_embeddings = self._get_closet_item_embeddings(user_id)
        closet_items = create_closet_items(closet, item_embeddings)
        closet = Closet(closet_items)
        query_items = closet.generate_query_items()
        outfits_by_occasion = {}
        if not query_items:
            return outfits_by_occasion
        for occasion in occasions:
            outfits = self.generate_outfits_for_occasion(query_items, closet, self.model)
            if outfits:
                outfits_by_occasion[occasion] = outfits
        return outfits_by_occasion

    def generate_outfits_for_occasion(self, query_items: list, closet: Closet, model: Module):
        outfits = []
        for item in query_items:
            candidates = generate_outfit_item_candidates(
                item, query_items, closet)
            if not candidates.contains_sufficent_items():
                warning_msg = f"Cannot generate outfits for {occasion}, insufficient items"
                app.logger.warning(warning_msg)
                continue
            try:
                generated_outfits = generate_outfit_set_for_item(
                    item, candidates, model)
                outfits += serialize_outfits(generated_outfits)
            except OutfitGenerationError as e:
                app.logger.warning(f'Error generating outfits {e}')
                continue
        return outfits

    def _get_closet_item_embeddings(self, user_id: int):
        item_embeddings = self.embedding_service.get_closet_item_embeddings(
            user_id)
        return item_embeddings



def aggregate_embeddings(closet, occasion):
    items = closet.get_items_by_occasion(occasion)
    embeddings = torch.stack([item.embedding for item in items])
    normalized_embeddings = torch.nn.functional.normalize(embeddings,
                                                          p=2,
                                                          dim=1)
    return normalized_embeddings


def serialize_outfits(outfit_set: list):
    """Convert all items in outfit set to their respective id's"""
    for i, outfit_data in enumerate(outfit_set):
        outfit_set[i]['outfit'] = {
            category: item.item_id
            for category, item in outfit_data['outfit'].items()
        }
    return outfit_set


def generate_outfit_set_for_item(item, candidates, model, num_outfits=float('inf')):
    """ Generates a set of outfits that all include the item_count_by_category

        Parameters
        ----------
        item : Item
        Query item that must be included in all outfits
        candidates: OutfitItemCandidates
        candidate items that can be included in the outfits. All clothing articles in a
        user's closet that are for a query occasion

        Returns
        -------
        outfits : list
        List of outfit dictionaries
        """
    if not model:
        model = load_bilstm_vse()
    num_outfits_generated = 0
    outfits = []
    able_to_generate_outfits = True
    occasion_candidates = OutfitItemCandidates(candidates.items)
    embeddings_for_occasion = occasion_candidates.compute_normalized_embeddings(
    )
    while able_to_generate_outfits and num_outfits_generated < num_outfits:
        app.logger.info(f'Generating {num_outfits_generated}th outfit')
        candidate_indeces = get_candidate_indeces(candidates,
                                                  occasion_candidates)
        recommended_items, probabilities = recommend_outfit_items(
            item, candidates, model, embeddings_for_occasion,
            candidate_indeces)
        app.logger.info('Filetering outfits in set')
        filtered_outfit_items, item_probabilities = filter_items(
            recommended_items, probabilities, item)
        app.logger.info('Completing outfits in set')
        outfit, missing_item_probabilities = complete_outfit(
            filtered_outfit_items, candidates, model, occasion_candidates,
            embeddings_for_occasion)
        outfit_score = compute_compatibility(item_probabilities,
                                             missing_item_probabilities)
        if outfit_score < COMPATIBILITY_SCORE_THRESHOLD:
            break
        weather_category = assign_weather_label(outfit)
        outfit_data = {"outfit": outfit, "weather_category": weather_category}
        outfits.append(outfit_data)
        num_outfits_generated += 1
        app.logger.info(
            f'Finished generating outfit num: {num_outfits_generated}')
        # Candidate items change so next outfit contains different items
        try:
            app.logger.warning(
                f"Candidates before update:{len([can.item_id for can in candidates.items])}"
            )
            candidates.update_items(outfit, item)
            app.logger.warning(
                f"Candidates after update:{len([can.item_id for can in candidates.items])}"
            )
        except NotEnoughClosetItemsException:
            app.logger.warning(candidates.categories_count)
            able_to_generate_outfits = False
    return outfits


def compute_compatibility(item_probabilities, missing_item_probabilities):
    """ Compute the compatibility score of an outfit
    
        Parameters
        ----------
        item_probabilities : list
        probabilities of each item occuring in an outfit
        missing_item_probabilities: list
        probabilities of each item that had been resolved, occuring in an outfit

        Returns
        -------
        compatibiltiy score : float
        """
    probabilities = item_probabilities + missing_item_probabilities
    return np.mean(probabilities)


def recommend_outfit_items(first_item: Item, candidates: OutfitItemCandidates,
                           model, embeddings_for_occasion,
                           candidate_indeces) -> list:
    first_item_embedding = first_item.embedding.unsqueeze(0)
    forward_seq, forward_probs = run_lstm(first_item_embedding,
                                          candidates, model,
                                          embeddings_for_occasion,
                                          candidate_indeces,
                                          LSTMDirection.FORWARD)
    backward_seq, backward_probs = run_lstm(first_item_embedding,
                                            candidates, model,
                                            embeddings_for_occasion,
                                            candidate_indeces,
                                            LSTMDirection.BACKWARD)
    outfit_items = backward_seq + forward_seq
    outfit_item_probabilities = forward_probs + backward_probs
    return outfit_items, outfit_item_probabilities


def get_candidate_indeces(candidates: OutfitItemCandidates,
                          occasion_candidates: OutfitItemCandidates) -> list:
    """ Get the index of each candidate in the list of candidates for an occasion

        Parameters
        ----------
        candidates : OutfitItemCandidates
        occasion_candidates: OutfitItemCandidates

        Returns
        -------
        item_indeces: list
    """
    candidate_ids = [item.item_id for item in candidates.items]
    item_ids = [item.item_id for item in occasion_candidates.items]
    item_indeces = [
        item_ids.index(candidate_id) for candidate_id in candidate_ids
    ]
    return item_indeces


def filter_items(recommended_outfit_items: list, probabilities: list,
                 query_item: Item) -> list:
    """ Running the model against candidates can sometimes produce sequences
    that have the same item twice so this function ensures there's only one item
    set per  category and it removes unnecessary categories

        Parameters
        ----------
        recommended_outfit_items : list
        items recommended by the model
        probabilities: list
        probability of each recommended item of being part of the outfit
        query_item: Item
        item that must be included in each outfit

        Returns
        -------
        filtered_outfit: dict
        outfit with no duplicate items and no unnecessary categories
        filtered_item_probabilities: list
        probability of each set item being part of the outfit
    """
    unique_items, item_probabilities = set_item_per_category(
        recommended_outfit_items, probabilities)
    unique_items[query_item.category] = query_item
    filtered_outfit, filtered_item_probabilities = remove_unecessary_items(
        unique_items, item_probabilities, query_item.category)
    return filtered_outfit, list(filtered_item_probabilities.values())


def set_item_per_category(recommended_outfit_items, probabilities):
    """ Set item with highest probability as the item for an outfit category

        Parameters
        ----------
        recommended_outfit_items : list
        items recommended by the modeprobabilities: list
        probability of each recommended item of being part of the outfit
        query_item: Item
        item that must be included in each outfit

        Returns
        -------
        unique_items: dict
        max_prob_per_category: dict
    """
    unique_items = {
        JACKET_CATEGORY_NAME: None,
        ONE_PIECE_CATEGORY_NAME: None,
        TOP_CATEGORY_NAME: None,
        BOTTOM_CATEGORY_NAME: None,
        SHOE_CATEGORY_NAME: None,
        ACCESSORY_CATEGORY_NAME: None
    }
    max_prob_per_category = {
        JACKET_CATEGORY_NAME: 0,
        ONE_PIECE_CATEGORY_NAME: 0,
        TOP_CATEGORY_NAME: 0,
        BOTTOM_CATEGORY_NAME: 0,
        SHOE_CATEGORY_NAME: 0,
        ACCESSORY_CATEGORY_NAME: 0
    }
    for idx, item in enumerate(recommended_outfit_items):
        item_probability = probabilities[idx]
        if item_probability > max_prob_per_category[item.category]:
            max_prob_per_category[item.category] = item_probability
            unique_items[item.category] = item
    return unique_items, max_prob_per_category


def remove_unecessary_items(outfit, item_probabilities, query_item_category):
    if outfit[ONE_PIECE_CATEGORY_NAME]:
        if query_item_category in [TOP_CATEGORY_NAME, BOTTOM_CATEGORY_NAME]:
            outfit.pop(ONE_PIECE_CATEGORY_NAME)
            item_probabilities.pop(ONE_PIECE_CATEGORY_NAME)
        else:
            outfit.pop(TOP_CATEGORY_NAME)
            item_probabilities.pop(TOP_CATEGORY_NAME)
            outfit.pop(BOTTOM_CATEGORY_NAME)
            item_probabilities.pop(BOTTOM_CATEGORY_NAME)
    else:
        if query_item_category == ONE_PIECE_CATEGORY_NAME:
            outfit.pop(TOP_CATEGORY_NAME)
            item_probabilities.pop(TOP_CATEGORY_NAME)
            outfit.pop(BOTTOM_CATEGORY_NAME)
            item_probabilities.pop(BOTTOM_CATEGORY_NAME)
        else:
            outfit.pop(ONE_PIECE_CATEGORY_NAME)
            item_probabilities.pop(ONE_PIECE_CATEGORY_NAME)
    return outfit, item_probabilities


def complete_outfit(outfit, candidates, model, occasion_candidates,
                    embeddings_for_occasion):
    """ Complet outfit by running model against missing categories and possible items

        Parameters
        ----------
        outfit: dict
        candidates: OutfitItemCandidates,
        item candidates that have been filtered down to reduce duplicate outfits
        model: pytorch model object
        occasion_candidates: OutfitItemCandidates
        all item candidates for a particular occasion
        embeddings_for_occasion: list
        Normalizd embeddings for all item candidates for a particular occasion

        Returns
        -------
        outfit: dict
        Outfit where no category can be set to None
        max_item_probabilities: list
        List of probabilities for each item that was added to complete an outfit
    """
    missing_categories = {}
    missing_item_probabilities = []
    essential_categories = [
        ONE_PIECE_CATEGORY_NAME, TOP_CATEGORY_NAME, BOTTOM_CATEGORY_NAME
    ]
    nonessential_missing_categories = []
    for category, item in outfit.items():
        if not item:
            if category in essential_categories:
                items = candidates.get_items_by_category(category)
                if not items:
                    raise OutfitGenerationError(
                        "Insufficient candidate items to complete outfit")
                missing_categories[category] = OutfitItemCandidates(items)
            else:
                nonessential_missing_categories.append(category)

    # Remove non-essential missing categories from outfit
    [outfit.pop(category) for category in nonessential_missing_categories]

    if missing_categories:
        completed_outfit, missing_item_probabilities = resolve_missing_items(
            outfit, missing_categories, model, occasion_candidates,
            embeddings_for_occasion)
        return completed_outfit, missing_item_probabilities
    return outfit, missing_item_probabilities


def resolve_missing_items(outfit, missing_categories, model,
                          occasion_candidates, embeddings_for_occasion):

    missing_item_probabilities = []
    for missing_category, candidates in missing_categories.items():
        outfit_items = [
            item for category, item in outfit.items()
            if item or category == missing_category
        ]
        missing_item_idx = outfit_items.index(None)
        forward_seq = outfit_items[:missing_item_idx]
        backward_seq = outfit_items[len(outfit_items):missing_item_idx:-1]
        probabilities, indices = compute_candidate_probabilities(
            backward_seq, forward_seq, candidates, model, occasion_candidates,
            embeddings_for_occasion)
        max_probability = 0
        for i, probability in enumerate(probabilities):
            if probability > max_probability:
                max_probability = probability
                max_index = indices[i]
        outfit[missing_category] = occasion_candidates.get_item_by_index(
            max_index)
        missing_item_probabilities.append(max_probability)
    return outfit, missing_item_probabilities


def compute_candidate_probabilities(backward_seq: list, forward_seq: list,
                                    candidates: OutfitItemCandidates, model,
                                    occasion_candidates,
                                    embeddings_for_occasion):
    """ Complete outfit by running model against missing categories and possible items

        Parameters
        ----------
        backward_seq: list
        sequence of items before missing item
        forward_seq: list
        sequence of items after missing item
        candidates: OutfitItemCandidates,
        candidates that can be used in the outfit
        model: pytorch model object
        occasion_candidates: OutfitItemCandidates
        all item candidates for a particular occasion
        embeddings_for_occasion: list
        Normalizd embeddings for all item candidates for a particular occasion

        Returns
        -------
        max_probabilities: list
        Probabilitiies of the items that will be used in the outfit
        indeces: list
        Indeces of the occasion candidates corresponding to the items that will
        be used in the outfit
    """
    candidate_indeces = get_candidate_indeces(candidates, occasion_candidates)
    max_probabilities = []
    indices = []
    if backward_seq:
        max_probability, idx = predict_best_candidate(backward_seq,
                                                      embeddings_for_occasion,
                                                      LSTMDirection.BACKWARD, model,
                                                      candidate_indeces)
        max_probabilities.append(max_probability)
        indices.append(idx)
    if forward_seq:
        max_probability, idx = predict_best_candidate(forward_seq,
                                                      embeddings_for_occasion,
                                                      LSTMDirection.FORWARD, model,
                                                      candidate_indeces)
        max_probabilities.append(max_probability)
        indices.append(idx)
    return max_probabilities, indices


def predict_best_candidate(sequence, candidate_embeddings, direction, model,
                           candidate_indeces):
    sequence_embeddings = compute_normalized_embeddings(sequence)
    probability, idx = predict_single_direction(sequence_embeddings,
                                                candidate_embeddings,
                                                direction, model,
                                                candidate_indeces)
    return probability, idx


def compute_normalized_embeddings(item_seq):
    item_embeddings = [item.embedding for item in item_seq]
    if not item_embeddings:
        app.logger.warning('No item embeddings to normalize')
    item_embeddings = torch.stack(item_embeddings)
    normalized_embeddings = torch.nn.functional.normalize(item_embeddings,
                                                          p=2,
                                                          dim=1)
    return normalized_embeddings


def load_outfits_for_user(gender, occasion):
    occasion = occasion.lower()
    gender = gender.lower()
    try:
        with open(PRE_GENERATED_OUTFITS_FILE_PATH) as f:
            outfits = json.load(f)
            return filter_loaded_outfits(gender, occasion, outfits)
    except FileNotFoundError as error:
        error_message = "Pregenerated outfits file not found in path {PRE_GENERATED_OUTFITS_FILE_PATH}"
        raise OutfitRetrievalException(error_message)


def get_outfit_ids(gender, occasion):
    with open(OUTFITS_START_END_IDS_FILE) as f:
        outfits_start_end_ids = json.load(f)
    try:
        start_index, end_index = outfits_start_end_ids[gender][occasion]
    except KeyError as error:
        error_message = f"No pregenerated outfit ids for gender: {gender} and occasion: {occasion}"
        raise OutfitRetrievalException(error_message)
    return [str(index) for index in range(start_index, end_index)]


def filter_loaded_outfits(gender, occasion, loaded_outfits):
    outfit_ids = get_outfit_ids(gender, occasion)
    filtered_outfits = {}
    for outfit_id in outfit_ids:
        try:
            filtered_outfits[outfit_id] = loaded_outfits[outfit_id]
        except KeyError as error:
            error_message = f"Outfit id for gender/occasion not found in loaded outfits, outfit_id: {outfit_id}"
            raise OutfitRetrievalException(error_message)
    return filtered_outfits


def map_basic_ids_to_item_ids(outfit, stock_image_item_id_map):
    mapped_outfits = {}
    for category, stock_image_id in outfit.items():
        try:
            mapped_outfits[category] = stock_image_item_id_map[str(
                stock_image_id)]
        except KeyError as error:
            error_message = f"No item id maps to outfit stock image id {stock_image_id}"
            raise OutfitRetrievalException(error_message)
    return mapped_outfits


def create_closet_items(closet, item_embeddings):
    items = []
    for category in closet:
        for item in category['items']:
            embedding = torch.Tensor(item_embeddings[str(item['item_id'])])
            category = item['category_name']
            type_name = item['type_name']
            image_url = item['image_url']
            item = Item(item['item_id'], item['occasion_tags'], embedding,
                        category, image_url, type_name)
            items.append(item)
    return items


def generate_outfit_item_candidates(query_item: Item, query_items: list, closet: Closet):
    items_to_exclude = query_items.remove(query_item)
    candidates = list(set(closet.items) - set(query_items))
    return OutfitItemCandidates(candidates)


def assign_weather_label(outfit):
    '''Assign a weather category for which an outfit can be worn
    based on the outfit items' types.'''
    outfit_item_types = np.array([
        WEATHER_MAPPINGS.get(item.type_name) for item in outfit.values()
        if item.type_name and WEATHER_MAPPINGS.get(item.type_name)
    ])
    if np.all(outfit_item_types == "hot"):
        return "hot"
    elif np.all(outfit_item_types == "cold"):
        return "cold"
    else:
        return "neutral"
