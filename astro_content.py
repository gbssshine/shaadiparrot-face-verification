"""Mithu's question of the day: one light question about family and life for everyone in India.
The same question for the whole day; the list rotates by the day number."""
from __future__ import annotations

from datetime import date as Date
from typing import Any, Dict, List

QUESTIONS: List[Dict[str, Any]] = [
    {"id": "wakes_first", "text": "Who wakes up first in your home?", "options": ["Me", "My partner", "The kids", "Mummy-ji"]},
    {"id": "arranged_love", "text": "Arranged or love marriage: which one lasts longer?", "options": ["Arranged", "Love", "Both the same"]},
    {"id": "remote", "text": "Who holds the TV remote in your house?", "options": ["Me", "My partner", "The kids", "The elders"]},
    {"id": "mom_calls", "text": "How many times a day does your mom call?", "options": ["Once", "2–3 times", "5 or more", "She just visits"]},
    {"id": "sunday", "text": "The perfect Sunday is…", "options": ["A long sleep", "Family lunch", "Going out", "Finishing chores"]},
    {"id": "fight_topic", "text": "The most common argument at home is about…", "options": ["Money", "Relatives", "Screen time", "Food"]},
    {"id": "dinner_decider", "text": "Who decides what's for dinner?", "options": ["Me", "My partner", "Mummy", "Whoever is hungriest"]},
    {"id": "kundli_check", "text": "Should families match kundlis before marriage?", "options": ["Always", "Only if elders want", "Not needed"]},
    {"id": "make_up", "text": "How do you make up after a fight?", "options": ["Food", "A sorry message", "Silence, then normal", "A hug"]},
    {"id": "marry_age", "text": "The right age to get married is…", "options": ["Before 25", "25–28", "28–32", "Whenever it feels right"]},
    {"id": "joint_family", "text": "Joint family or nuclear family?", "options": ["Joint", "Nuclear", "Nearby but separate"]},
    {"id": "best_cook", "text": "Who is the best cook in your family?", "options": ["Me", "Mom", "My partner", "Grandma"]},
    {"id": "getaway", "text": "A long weekend: go somewhere or stay home?", "options": ["Go somewhere", "Stay home"]},
    {"id": "phone_password", "text": "Should partners know each other's phone password?", "options": ["Yes", "No", "They know anyway"]},
    {"id": "morning_first", "text": "The first thing you do in the morning?", "options": ["Check my phone", "Pray or meditate", "Exercise", "Wake the kids"]},
    {"id": "festival_food", "text": "The best part of festival food is…", "options": ["Sweets", "Chaat", "Biryani", "Mom's thali"]},
    {"id": "anniversary", "text": "How do you celebrate anniversaries?", "options": ["Dinner out", "A gift", "A prayer or puja together", "We forget"]},
    {"id": "trip_planner", "text": "Who plans the family trips?", "options": ["Me", "My partner", "The elders", "It just happens"]},
    {"id": "kids_screens", "text": "How much screen time is fine for kids?", "options": ["Under 1 hour", "1–2 hours", "More is fine", "No screens"]},
    {"id": "match_matters", "text": "What matters most in a match?", "options": ["Values", "Family", "Career", "Kundli"]},
    {"id": "first_sight", "text": "Do you believe in love at first sight?", "options": ["Yes", "No", "Maybe"]},
    {"id": "family_group", "text": "Your family WhatsApp group is…", "options": ["Good-morning flowers", "Pure chaos", "Muted", "Actually useful"]},
    {"id": "rishta_source", "text": "The best rishtas come from…", "options": ["Relatives", "Apps", "Friends"]},
    {"id": "saver_spender", "text": "At home, who saves and who spends?", "options": ["I save", "I spend", "We both save", "We both spend"]},
    {"id": "gift_gold", "text": "The best wedding gift is…", "options": ["Gold", "Cash", "Something for the home", "A trip"]},
    {"id": "dinner_time", "text": "Family dinner happens…", "options": ["Before 8", "8 to 9", "After 9", "Whenever"]},
    {"id": "read_horoscope", "text": "How often do you read your horoscope?", "options": ["Every day", "Sometimes", "Only before big days"]},
    {"id": "sorry_first", "text": "Who says sorry first in your home?", "options": ["Me", "My partner", "Whoever was wrong", "Nobody"]},
    {"id": "movie_night", "text": "Family movie night pick?", "options": ["Comedy", "Old Bollywood", "Cartoons", "Cricket instead"]},
    {"id": "kids_count", "text": "The ideal number of kids?", "options": ["One", "Two", "Three or more", "Not sure"]},
    {"id": "move_city", "text": "Would you move cities for your partner's job?", "options": ["Yes", "No", "Depends"]},
    {"id": "walk", "text": "Morning walk or evening walk?", "options": ["Morning", "Evening", "Neither"]},
    {"id": "birthdays", "text": "Who remembers every birthday in your family?", "options": ["Me", "Mom", "My partner", "Phone reminders"]},
    {"id": "holiday_spot", "text": "Your family's favourite holiday?", "options": ["Hills", "Beach", "The village home", "A pilgrimage"]},
    {"id": "vrat", "text": "Do you keep fasts (vrat, roza, upvas)?", "options": ["Regularly", "On festivals", "Never"]},
    {"id": "wedding_size", "text": "The ideal wedding size?", "options": ["Under 100 guests", "100–300", "500+", "Court marriage"]},
    {"id": "pocket_money", "text": "How should kids get their money?", "options": ["Pocket money", "Earn it with chores", "Both"]},
    {"id": "wins_arguments", "text": "Who wins the arguments at home?", "options": ["Me", "My partner", "Mummy", "The kids"]},
    {"id": "festival_day", "text": "The best way to spend a festival day?", "options": ["With family", "With friends", "Travelling", "Resting"]},
    {"id": "inlaws_visit", "text": "How often should in-laws visit?", "options": ["Every week", "Every month", "Festivals only", "They live with us"]},
    {"id": "love_language", "text": "How does your family show love?", "options": ["Food", "Worry", "Gifts", "Time together"]},
    {"id": "late_night", "text": "Who stays up the latest in your home?", "options": ["Me", "My partner", "The kids", "Everyone"]},
]


def question_for(day: Date) -> Dict[str, Any]:
    return QUESTIONS[day.toordinal() % len(QUESTIONS)]
