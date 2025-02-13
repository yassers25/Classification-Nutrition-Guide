import streamlit as st
import sys
import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from pathlib import Path
import subprocess

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_path)

from predict import TripleModelPredictionPipeline
from config import Config

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent  # Remontez au répertoire racine du projet

# Ajout de la fonction pour gérer les chemins
def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on current file location"""
    current_dir = Path(__file__).parent.parent.parent  # Remontez au répertoire racine du projet
    return str(current_dir / relative_path)

PRODUCT_INFO = {
    'apple': {
        'calories': 'Around 95 calories per medium apple (182g)',
        'nutrients': '''Rich in fiber (4.5g), vitamin C (14% DV), and potassium. Contains beneficial antioxidants like quercetin and catechin. Good source of soluble fiber called pectin.''',
        'taste': '''Sweet with a balanced tartness. Texture ranges from crisp to tender depending on variety. Fresh, clean flavor with subtle floral notes.''',
        'benefits': '''• Supports heart health: Apples contain flavonoids and antioxidants, which may help reduce blood pressure and LDL cholesterol, protecting against cardiovascular diseases.
        \n• Helps regulate blood sugar: The soluble fiber in apples slows the digestion of sugars and improves blood sugar levels, making them beneficial for people with diabetes.
        \n• Promotes good gut bacteria: Pectin, a type of soluble fiber in apples, acts as a prebiotic, feeding beneficial gut bacteria and improving overall gut health.
        \n• May reduce risk of certain cancers: The antioxidants and plant compounds in apples have been linked to a reduced risk of cancers such as colorectal, breast, and lung cancer.
        \n• Supports immune system: Apples contain vitamin C and quercetin, which strengthen the immune system by reducing inflammation and supporting white blood cell function.'''
        },
        'banana': {
    'calories': 'About 105 calories per medium banana (118g)',
    'nutrients': '''High in potassium (422mg), vitamin B6 (33% DV), vitamin C, magnesium, and fiber (3.1g). Contains resistant starch that aids digestion.''',
    'taste': '''Naturally sweet with a creamy texture. Flavor intensifies as the fruit ripens, developing caramel-like notes.''',
    'benefits': '''• Excellent pre/post workout snack: Bananas provide quick-digesting carbs and electrolytes like potassium, making them ideal for replenishing energy and preventing cramps during workouts.
    \n• Supports heart health: High potassium and low sodium content help regulate blood pressure and reduce the risk of cardiovascular diseases.
    \n• Aids digestion: Rich in fiber and resistant starch, bananas promote healthy digestion and regular bowel movements while soothing the stomach.
    \n• Helps regulate blood sugar: The soluble fiber in bananas slows sugar absorption, helping maintain stable blood sugar levels, particularly when eaten with a protein or fat source.
    \n• Natural mood booster: Bananas contain tryptophan, a precursor to serotonin, which supports mood regulation and can help reduce stress and anxiety.'''
    },
    'beetroot': {
        'calories': 'About 43 calories per 100g',
        'nutrients': '''High in fiber, folate (20% DV), manganese, potassium, and vitamin C. Contains unique antioxidants called betalains.''',
        'taste': '''Earthy, sweet flavor with a subtle mineral undertone. Tender and juicy texture when cooked.''',
        'benefits': '''• Improves athletic performance: Rich in dietary nitrates, beetroot enhances oxygen efficiency and stamina, making it a favorite among athletes.
         \n• Supports blood pressure regulation: Dietary nitrates in beetroot convert to nitric oxide, helping relax blood vessels and lower blood pressure.
         \n• Anti-inflammatory properties: Contains betalains, which help reduce inflammation and may lower the risk of chronic diseases.
         \n• Supports liver health: Beetroot aids detoxification through compounds like betaine, which supports liver function.
         \n• Rich in antioxidants: Betalains and vitamin C protect cells from oxidative stress, reducing damage caused by free radicals.'''
    },
    'bell pepper': {
        'calories': 'About 30 calories per medium pepper (119g)',
        'nutrients': '''Excellent source of vitamin C (169% DV), vitamin A, potassium, and folate. Contains antioxidants like capsanthin.''',
        'taste': '''Sweet and crisp with varying levels of sweetness depending on color. Green ones are slightly bitter.''',
        'benefits': '''• Supports eye health: High levels of vitamin A and carotenoids like lutein and zeaxanthin promote good vision and reduce the risk of eye diseases.
        \n• Boosts immune system: The exceptionally high vitamin C content enhances immune function and helps fight infections.
        \n• Promotes healthy skin: Vitamin C supports collagen production, which is vital for skin elasticity and repair.
        \n• Aids in iron absorption: Vitamin C enhances the absorption of non-heme iron, making it beneficial for preventing anemia.
        \n• Anti-inflammatory properties: Contains antioxidants and phytonutrients that reduce inflammation and support overall health.'''
    },
    'cabbage': {
        'calories': 'About 17 calories per cup (89g)',
        'nutrients': '''High in vitamin K, vitamin C, fiber, and folate. Contains beneficial compounds like glucosinolates.''',
        'taste': '''Mild, slightly sweet when raw, becomes sweeter when cooked. Crisp texture when raw.''',
        'benefits': '''• Supports digestive health: Rich in fiber, cabbage promotes gut health and regularity while feeding beneficial gut bacteria.
        \n• Anti-inflammatory properties: Glucosinolates and antioxidants in cabbage reduce inflammation, benefiting chronic conditions.
        \n• May help reduce risk of certain cancers: Contains glucosinolates, which convert to compounds that may protect against cancer.
        \n• Supports heart health: Potassium and anthocyanins (especially in red cabbage) help maintain blood pressure and reduce cardiovascular risks.
        \n• Rich in antioxidants: Provides vitamin C and phytonutrients that combat oxidative stress, improving overall cellular health.'''
    },
    'capsicum': {
    'calories': 'About 31 calories per 100g',
    'nutrients': '''Rich in vitamins A, C, and E. Contains folate, fiber, and various antioxidants like carotenoids (including beta-carotene).''',
    'taste': '''Sweet and mildly tangy, with a crisp, juicy texture. The flavor can vary depending on color (red, yellow, green, or orange).''',
    'benefits': '''• Supports eye health: High vitamin A content promotes healthy vision.
    \n• Boosts immune system: Vitamin C helps fight infections and strengthens immunity.
    \n• Anti-inflammatory properties: Antioxidants reduce inflammation and protect against chronic disease.
    \n• Promotes healthy skin: Vitamin C supports collagen production for healthy, glowing skin.
    \n• Aids digestion: The fiber content aids in maintaining a healthy digestive system.'''
},
    'carrot': {
        'calories': 'About 41 calories per 100g',
        'nutrients': '''Excellent source of vitamin A (beta carotene), fiber, potassium, and antioxidants. Contains biotin and vitamin K1.''',
        'taste': '''Sweet and crunchy when raw, becomes sweeter when cooked. Earthy undertones.''',
        'benefits': '''• Promotes eye health: High levels of beta-carotene convert to vitamin A, essential for good vision and preventing night blindness.
        \n• Supports immune function: Antioxidants and vitamin A help strengthen the immune system.
        \n• Helps maintain healthy skin: Vitamin A promotes skin repair and prevents dryness.
        \n• Good for heart health: Potassium helps regulate blood pressure, and antioxidants reduce the risk of cardiovascular diseases.
        \n• Supports dental health: Crunchy texture stimulates gums and helps clean teeth, acting as a natural toothbrush.'''
    },
    'cauliflower': {
        'calories': 'About 25 calories per cup (100g)',
        'nutrients': '''High in fiber, vitamins C, K, and B6. Good source of folate and pantothenic acid.''',
        'taste': '''Mild, slightly nutty flavor that becomes sweeter when cooked. Tender-crisp texture.''',
        'benefits': '''• Supports brain health: High levels of choline improve cognitive function and memory.
        \n• Anti-inflammatory properties: Contains antioxidants that reduce inflammation and support overall health.
        \n• Good for weight management: Low in calories and high in fiber, helping with satiety and healthy digestion.
        \n• Supports detoxification: Contains compounds like sulforaphane that aid liver detoxification.
        \n• Rich in choline: Essential for cell membrane structure and neurotransmitter production.'''
    },
    'chili pepper': {
    'calories': 'About 40 calories per 100g',
    'nutrients': '''High in vitamin C, vitamin A, potassium, and capsaicin (which gives the heat). Good source of fiber and antioxidants. Contains small amounts of vitamin B6, folate, and vitamin K.''',
    'taste': '''Spicy and pungent, with a sharp heat that varies depending on the variety. The flavor can be smoky, sweet, or earthy depending on the type of chili.''',
    'benefits': '''• Boosts metabolism: Capsaicin helps increase metabolism and may promote fat burning.
        \n• Reduces pain: Capsaicin has pain-relieving properties and is often used in topical creams.
        \n• Supports heart health: Antioxidants like vitamin C and capsaicin promote heart health.
        \n• Aids digestion: Chili peppers stimulate digestive enzymes and improve gut health.
        \n• May help with weight loss: Capsaicin can help suppress appetite and increase calorie burning.'''
},

'corn': {
    'calories': 'About 96 calories per 100g (cooked)',
    'nutrients': '''Good source of fiber, vitamin C, B vitamins (especially B5 and B9), and folate. Contains antioxidants such as lutein and zeaxanthin.''',
    'taste': '''Sweet, slightly starchy flavor with a crunchy texture when fresh or cooked. The flavor can vary depending on the variety, with some being sweeter than others.''',
    'benefits': '''• Supports eye health: Antioxidants like lutein and zeaxanthin help protect vision.
        \n• Aids digestion: High fiber content promotes healthy digestion and regular bowel movements.
        \n• Boosts energy: Complex carbohydrates provide a steady source of energy.
        \n• Supports heart health: Fiber and antioxidants help maintain a healthy heart.
        \n• Promotes healthy skin: Vitamin C supports skin regeneration and collagen formation.'''
},
    'cucumber': {
        'calories': 'About 8 calories per 100g',
        'nutrients': '''High water content (96%). Contains vitamin K, potassium, and magnesium. Good source of antioxidants.''',
        'taste': '''Cool, crisp, and refreshing with a mild, slightly sweet flavor.''',
        'benefits': '''• Hydrating properties: High water content keeps the body hydrated and aids temperature regulation.
        \n• Supports digestive health: Fiber and water content help prevent constipation and promote gut health.
        \n• Good for weight management: Low in calories and filling, making it a great snack for weight loss.
        \n• Helps reduce inflammation: Antioxidants like cucurbitacins combat inflammation and oxidative stress.
        \n• Supports skin health: Hydration and silica content improve skin elasticity and reduce puffiness.'''
    },
    'eggplant': {
        'calories': 'About 20 calories per cup (82g)',
        'nutrients': '''Good source of fiber, potassium, vitamin B1, and copper. Contains unique antioxidants like nasunin.''',
        'taste': '''Mild, slightly sweet flavor when cooked. Can be slightly bitter when raw. Meaty texture.''',
        'benefits': '''• Supports heart health: Antioxidants like nasunin help protect blood vessels and reduce cholesterol levels.
        \n• Rich in antioxidants: Protects cells from oxidative stress, reducing the risk of chronic diseases.
        \n• Helps with blood sugar control: Low in carbs and high in fiber, aiding in blood sugar management.
        \n• Supports brain function: Nasunin helps protect brain cells from free radical damage.
        \n• Good for digestion: High fiber content promotes regular bowel movements and a healthy gut microbiome.'''
    },
    'garlic': {
        'calories': 'About 4 calories per clove',
        'nutrients': '''Rich in manganese, vitamin B6, vitamin C, and selenium. Contains beneficial compounds like allicin.''',
        'taste': '''Strong, pungent flavor when raw, becomes sweeter and milder when cooked.''',
        'benefits': '''• Boosts immune system: Contains compounds that enhance immune response to infections.
        \n• Natural antibiotic properties: Allicin has antibacterial and antiviral effects.
        \n• Supports heart health: Helps lower cholesterol and blood pressure, reducing cardiovascular risk.
        \n• Anti-inflammatory effects: Reduces inflammation, benefiting conditions like arthritis.
        \n• May help reduce blood pressure: Compounds like allicin promote vasodilation and blood flow.'''
    },
    'ginger': {
        'calories': 'About 80 calories per 100g',
        'nutrients': '''Contains gingerols, shogaols, and zingerone. Good source of potassium, magnesium, and vitamin B6.''',
        'taste': '''Spicy, warm, and aromatic with a slight sweetness. Pungent when fresh.''',
        'benefits': '''• Aids digestion: Stimulates digestive enzymes and relieves discomfort from bloating or indigestion.
        \n• Reduces nausea: Effective for motion sickness, morning sickness, and post-surgery nausea.
        \n• Anti-inflammatory properties: Contains compounds like gingerol that reduce inflammation in conditions like arthritis.
        \n• Supports immune system: Antioxidants and antimicrobial properties help fight infections.
        \n• May help with pain relief: Known to ease muscle soreness, menstrual pain, and chronic conditions.'''
    },
    'grapes': {
        'calories': 'About 69 calories per cup (100g)',
        'nutrients': '''Contains resveratrol, vitamin K, and potassium. Good source of various antioxidants.''',
        'taste': '''Sweet and juicy with varying tartness depending on variety.''',
        'benefits': '''• Heart health benefits: Resveratrol and flavonoids help improve cholesterol levels and reduce heart disease risk.
        \n• Anti-aging properties: Antioxidants combat free radicals, promoting healthier skin and delaying aging.
        \n• Supports brain function: Resveratrol enhances memory and protects against neurodegenerative diseases.
        \n• May help with inflammation: Polyphenols reduce inflammation and oxidative stress in the body.
        \n• Contains cancer-fighting compounds: Antioxidants like resveratrol inhibit the growth of certain cancer cells.'''
    },
    'jalepeno': {
    'calories': 'About 4 calories per medium jalapeño (14g)',
    'nutrients': '''Good source of vitamin C, vitamin A, and B vitamins (especially B6). Contains capsaicin, which provides the heat. Also provides small amounts of fiber and potassium.''',
    'taste': '''Spicy, with a distinct sharp heat. Mildly smoky, grassy, and slightly sweet flavor when eaten raw, with a more intense heat when cooked.''',
    'benefits': '''• Boosts metabolism: Capsaicin helps increase metabolism and may aid in weight loss.
    \n• Rich in antioxidants: Helps combat free radicals, supporting overall health.
    \n• Supports heart health: The capsaicin and antioxidants may promote heart health.
    \n• Anti-inflammatory properties: Helps reduce inflammation in the body.
    \n• Improves digestion: Capsaicin stimulates digestive enzymes and promotes better gut health.'''
},
    'kiwi': {
        'calories': 'About 61 calories per 100g',
        'nutrients': '''High in vitamin C, vitamin K, vitamin E, folate, and potassium. Good source of fiber.''',
        'taste': '''Sweet-tart flavor with tropical notes. Soft, creamy texture with crunchy seeds.''',
        'benefits': '''• Supports digestive health: High fiber content and actinidin enzyme promote regularity and reduce bloating.
        \n• Boosts immune system: Rich in vitamin C, aiding in faster recovery from illnesses.
        \n• Improves sleep quality: Contains serotonin, which may help regulate sleep cycles.
        \n• Good for skin health: Vitamin C and antioxidants promote collagen production and reduce wrinkles.
        \n• Supports respiratory health: Helps alleviate asthma symptoms and reduces respiratory inflammation.'''
    },
    'lemon': {
        'calories': 'About 29 calories per 100g',
        'nutrients': '''Excellent source of vitamin C, citric acid, and flavonoids. Contains potassium and vitamin B6.''',
        'taste': '''Sour and acidic with bright, citrusy notes.''',
        'benefits': '''• Supports immune system: High vitamin C levels strengthen immune defense and fight colds.
        \n• Aids digestion: Citric acid stimulates digestive juices, helping nutrient absorption.
        \n• Helps with kidney stones: Citric acid prevents the formation of kidney stones by increasing citrate levels in urine.
        \n• Supports heart health: Antioxidants improve cholesterol levels and reduce heart disease risks.
        \n• Helps with iron absorption: Vitamin C enhances absorption of iron from plant-based foods.'''
    },
    'lettuce': {
        'calories': 'About 5 calories per cup',
        'nutrients': '''Good source of vitamin K, vitamin A, and folate. High water content and fiber.''',
        'taste': '''Mild, crisp, and refreshing. Slightly sweet with a subtle bitterness.''',
        'benefits': '''• Supports hydration: High water content helps maintain hydration levels.
        \n• Good for eye health: Rich in vitamin A, supporting good vision and preventing dryness.
        \n• Aids in weight management: Low calorie content makes it ideal for weight loss diets.
        \n• Promotes healthy sleep: Lactucarium in lettuce has mild sedative properties that improve sleep quality.
        \n• Supports bone health: Vitamin K is essential for bone density and reducing fracture risks.'''
    },
    'mango': {
        'calories': 'About 99 calories per cup',
        'nutrients': '''Rich in vitamins A and C, fiber, and antioxidants. Good source of copper and folate.''',
        'taste': '''Sweet and tropical with complex flavor notes. Smooth, creamy texture when ripe.''',
        'benefits': '''• Supports immune system: High in vitamin C, helping to combat infections and boost immunity.
        \n• Promotes eye health: Vitamin A and beta-carotene help improve vision and prevent night blindness.
        \n• Aids digestion: Contains digestive enzymes like amylase and fiber to improve gut health.
        \n• Supports skin health: Antioxidants and vitamins help reduce acne, improve skin elasticity, and combat aging.
        \n• Anti-inflammatory properties: Reduces inflammation, aiding in overall health and recovery.'''
    },
    'onion': {
        'calories': 'About 40 calories per 100g',
        'nutrients': '''Rich in quercetin, sulfur compounds, and vitamin C. Good source of B vitamins and potassium.''',
        'taste': '''Sharp and pungent when raw, sweet and savory when cooked.''',
        'benefits': '''• Anti-inflammatory properties: Quercetin and sulfur compounds help reduce inflammation in the body.
        \n• Supports heart health: Helps lower cholesterol levels and improve blood pressure regulation.
        \n• Antimicrobial effects: Contains compounds that combat harmful bacteria and viruses.
        \n• May help regulate blood sugar: Improves insulin function and stabilizes blood sugar levels.
        \n• Supports bone health: Rich in nutrients that contribute to bone density and strength.'''
    },
    'orange': {
        'calories': 'About 62 calories per medium orange',
        'nutrients': '''High in vitamin C, fiber, thiamine, folate, and potassium. Contains various antioxidants.''',
        'taste': '''Sweet and citrusy with varying levels of tartness. Juicy texture.''',
        'benefits': '''• Boosts immune system: Vitamin C strengthens immunity and helps prevent common colds.
        \n• Supports skin health: Antioxidants and vitamin C promote collagen production and improve skin elasticity.
        \n• Helps with iron absorption: Enhances the absorption of non-heme iron from plant-based foods.
        \n• Anti-inflammatory properties: Reduces inflammation and combats oxidative stress.
        \n• Supports heart health: Potassium and fiber help regulate blood pressure and improve cholesterol levels.'''
    },
    'paprika': {
    'calories': 'About 19 calories per tablespoon (6g)',
    'nutrients': '''Rich in vitamin A, vitamin E, and vitamin C. Good source of antioxidants like capsaicin, lutein, and zeaxanthin. Contains small amounts of iron, potassium, and B vitamins.''',
    'taste': '''Sweet and smoky with mild heat, depending on the variety. Can also have a slightly bitter flavor, with complex earthy notes.''',
    'benefits': '''• Rich in antioxidants: Helps fight oxidative stress and supports overall health.
    \n• Supports eye health: High in vitamin A and carotenoids, which are essential for maintaining good vision.
    \n• Anti-inflammatory properties: Contains capsaicin, which can help reduce inflammation in the body.
    \n• Boosts metabolism: The capsaicin in paprika may promote fat burning and aid in weight loss.
    \n• Improves skin health: The vitamin A content helps support healthy skin and can reduce signs of aging.'''
},

    'pear': {
        'calories': 'About 101 calories per medium pear',
        'nutrients': '''Good source of fiber, vitamin C, vitamin K, and copper. Contains beneficial plant compounds.''',
        'taste': '''Sweet and mild with a soft, grainy texture. Juicy when ripe.''',
        'benefits': '''• Supports digestive health: High fiber content promotes regular bowel movements and gut health.
        \n• Good for heart health: Fiber and antioxidants reduce cholesterol levels and improve circulation.
        \n• Helps with inflammation: Polyphenols and antioxidants reduce inflammation throughout the body.
        \n• Supports bone health: Vitamin K and copper are essential for maintaining bone strength.
        \n• Anti-cancer properties: Plant compounds like flavonoids help prevent the growth of cancer cells.'''
    },
    'peas': {
    'calories': 'About 81 calories per cup (160g)',
    'nutrients': '''Good source of vitamins A, C, and K, as well as folate, iron, and manganese. High in fiber and protein, and contains antioxidants like flavonoids and carotenoids.''',
    'taste': '''Sweet and slightly starchy with a mild, fresh flavor. Crisp texture when fresh, tender when cooked.''',
    'benefits': '''• Supports digestive health: High fiber content helps maintain healthy digestion.
        \n• Good for heart health: Rich in antioxidants and nutrients that support cardiovascular function.
        \n• Aids in weight management: Low in calories but high in fiber and protein, helping to promote satiety.
        \n• Boosts immune system: Vitamin C and other antioxidants help protect the body from illness.
        \n• Supports bone health: Vitamin K and other minerals contribute to maintaining strong bones.'''
},
    'pineapple': {
        'calories': 'About 82 calories per cup',
        'nutrients': '''Rich in vitamin C, manganese, and bromelain. Good source of thiamin and vitamin B6.''',
        'taste': '''Sweet and tropical with tangy notes. Juicy and fibrous texture.''',
        'benefits': '''• Aids digestion: Bromelain helps break down proteins and improves digestive health.
        \n• Anti-inflammatory properties: Reduces inflammation and supports recovery from injuries.
        \n• Supports immune system: Vitamin C boosts immunity and protects against infections.
        \n• Helps with wound healing: Promotes collagen production for faster healing.
        \n• May help with arthritis: Bromelain reduces joint pain and inflammation.'''
    },
    'pomegranate': {
        'calories': 'About 234 calories per pomegranate',
        'nutrients': '''High in punicalagins, vitamin C, potassium, and fiber. Rich in antioxidants.''',
        'taste': '''Sweet-tart flavor with complex notes. Juicy seeds with crunchy texture.''',
        'benefits': '''• Powerful antioxidant properties: Punicalagins and other compounds combat oxidative stress.
        \n• Supports heart health: Improves cholesterol levels and blood pressure.
        \n• Anti-inflammatory effects: Helps reduce chronic inflammation throughout the body.
        \n• May help with arthritis: Reduces symptoms and supports joint health.
        \n• Supports exercise performance: Improves endurance and reduces muscle soreness.'''
    },
    'potato': {
        'calories': 'About 110 calories per medium potato',
        'nutrients': '''Good source of vitamin C, potassium, vitamin B6, and fiber. Contains resistant starch.''',
        'taste': '''Mild, starchy flavor that varies by variety. Fluffy when cooked properly.''',
        'benefits': '''• Provides sustained energy: Complex carbohydrates provide a steady source of energy.
        \n• Supports heart health: Potassium helps regulate blood pressure and supports heart function.
        \n• Good for digestive health: Resistant starch and fiber promote gut health and regularity.
        \n• Supports immune system: Vitamin C enhances immunity and overall health.
        \n• Contains antioxidants: Protects cells from damage and supports healthy aging.'''
    },
    'raddish': {
    'calories': 'About 16 calories per 100g',
    'nutrients': '''Rich in vitamin C, folate, potassium, and fiber. Contains antioxidants such as anthocyanins and glucosinolates.''',
    'taste': '''Sharp, peppery flavor with a crunchy texture. More intense when raw, milder when cooked.''',
    'benefits': '''• Supports digestive health: High fiber content promotes healthy digestion.
    \n• Helps with weight management: Low in calories and high in fiber, aiding in satiety.
    \n• Good for skin health: Vitamin C supports collagen production for healthy skin.
    \n• Anti-inflammatory properties: Contains compounds that help reduce inflammation.
    \n• Supports heart health: Potassium helps regulate blood pressure and supports heart function.'''
},

'soy beans': {
    'calories': 'About 173 calories per 100g',
    'nutrients': '''Rich in protein, fiber, vitamins B1, B2, and K, as well as minerals like iron, calcium, and magnesium. Contains phytonutrients like isoflavones.''',
    'taste': '''Mild, nutty flavor with a soft texture when cooked. Can be slightly earthy.''',
    'benefits': '''• Supports muscle growth: High-quality protein supports muscle repair and growth.
    \n• Promotes heart health: Isoflavones and fiber help lower cholesterol and support cardiovascular function.
    \n• Aids in weight management: High protein and fiber content contribute to fullness.
    \n• Good for bone health: Contains calcium and magnesium, important for maintaining strong bones.
    \n• Hormonal balance: Isoflavones may help balance hormones, especially in women.'''
},
    'spinach': {
        'calories': 'About 7 calories per cup',
        'nutrients': '''Rich in iron, vitamin K, vitamin A, vitamin C, and folate. High in antioxidants.''',
        'taste': '''Mild, slightly sweet when young, more mineral taste when mature.''',
        'benefits': '''• Supports eye health: Rich in lutein and zeaxanthin, which protect against macular degeneration.
        \n• Good for bone health: Vitamin K and calcium support strong bones.
        \n• Helps with blood pressure: Potassium and nitrates help maintain healthy blood pressure levels.
        \n• Anti-inflammatory properties: Reduces inflammation and supports overall health.
        \n• Supports brain function: Folate and antioxidants improve cognitive health and memory.'''
    },
    'sweetcorn': {
    'calories': 'About 96 calories per cup (cooked)',
    'nutrients': '''Rich in fiber, vitamin C, B vitamins (especially B5), potassium, and antioxidants like lutein and zeaxanthin.''',
    'taste': '''Sweet and slightly earthy with a chewy, crunchy texture. Flavor intensifies when roasted or grilled.''',
    'benefits': '''• Supports digestive health: High fiber content helps maintain a healthy digestive system.
    \n Boosts immune system: Vitamin C helps strengthen the immune response.
    \n Promotes eye health: Lutein and zeaxanthin are antioxidants that support eye health.
    \n Aids in heart health: Potassium helps regulate blood pressure and heart function.
    \n Good source of energy: Natural sugars and complex carbohydrates provide long-lasting energy.'''
},

'sweetpotato': {
    'calories': 'About 86 calories per 100g (baked)',
    'nutrients': '''Rich in vitamin A (from beta-carotene), vitamin C, fiber, potassium, and manganese. Contains antioxidants like anthocyanins.''',
    'taste': '''Sweet, earthy flavor with a soft, creamy texture when cooked. Slightly nutty taste when roasted.''',
    'benefits': '''• Supports eye health: High in beta-carotene, which supports vision and skin health.
    \n Aids digestion: High in fiber, which helps maintain digestive health and regulate bowel movements.
    \n Supports immune function: Vitamin C helps strengthen the immune system.
    \n Promotes heart health: Potassium helps manage blood pressure and support heart function.
    \n Anti-inflammatory properties: Contains compounds that help reduce inflammation.'''
},
    'tomato': {
        'calories': 'About 22 calories per medium tomato',
        'nutrients': '''Rich in lycopene, vitamin C, potassium, and vitamin K. Good source of fiber.''',
        'taste': '''Sweet-acidic balance with umami notes. Juicy texture.''',
        'benefits': '''• Supports heart health: Lycopene improves cholesterol levels and reduces heart disease risk.
    \n• Good for skin health: Vitamin C promotes collagen production and reduces UV damage.
    \n• Aids in cancer prevention: Lycopene and antioxidants combat oxidative stress and reduce cancer risk.
    \n• Supports bone health: Vitamin K and potassium contribute to strong bones.
    \n• Anti-inflammatory properties: Reduces inflammation and promotes overall well-being.'''
    },
    'turnip': {
    'calories': 'About 28 calories per 100g',
    'nutrients': '''Good source of vitamin C, potassium, folate, and fiber. Contains antioxidants such as glucosinolates and beta-carotene.''',
    'taste': '''Mildly sweet and slightly peppery with a crunchy texture when raw. Becomes tender and earthy when cooked.''',
    'benefits': '''• Supports digestive health: High fiber content promotes healthy digestion.
    \n• Boosts immune system: Vitamin C helps strengthen the immune response.
    \n• Aids in heart health: Potassium supports healthy blood pressure levels.
    \n• Anti-inflammatory properties: Glucosinolates and other compounds help reduce inflammation.
    \n• Supports weight management: Low in calories and high in fiber, making it a filling, nutritious option.'''
},
    'watermelon': {
        'calories': 'About 46 calories per cup',
        'nutrients': '''High in vitamin C, vitamin A, and lycopene. Good source of potassium and magnesium.''',
        'taste': '''Sweet and refreshing with high water content. Crisp, juicy texture.''',
        'benefits': '''• Supports hydration: High water content keeps you hydrated and refreshed.
    \n• Heart health benefits: Lycopene and potassium support cardiovascular health.
    \n• Reduces muscle soreness: Citrulline improves recovery and reduces soreness after exercise.
    \n• Anti-inflammatory properties: Reduces inflammation and oxidative stress.
    \n• Supports skin health: Vitamins A and C promote glowing, healthy skin.'''
}

}

class SegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def filter_boxes(self, boxes, scores, iou_threshold=0.5, score_threshold=0.7):
        """
        Filter boxes using Non-Maximum Suppression and additional criteria
        """
        # Convert to numpy for easier manipulation
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        # Filter by score first
        high_score_idx = scores_np > score_threshold
        boxes_np = boxes_np[high_score_idx]
        scores_np = scores_np[high_score_idx]
        
        # Calculate areas
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        
        # Sort boxes by score
        order = scores_np.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Get IoU between the current box and all remaining boxes
            xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
            yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
            xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
            yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Filter out boxes with high IoU
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return boxes_np[keep]

    def filter_global_box(self, boxes, image_size, area_threshold=0.7):
        """
        Filter out boxes that cover too much of the image
        """
        image_area = image_size[0] * image_size[1]
        filtered_boxes = []
        
        for box in boxes:
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area / image_area < area_threshold:
                filtered_boxes.append(box)
                
        return filtered_boxes

    def segment_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = F.to_tensor(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        
        # Apply NMS and score filtering
        filtered_boxes = self.filter_boxes(boxes, scores)
        
        # Filter out global boxes
        filtered_boxes = self.filter_global_box(filtered_boxes, image.size)
        
        # Create crops for remaining boxes
        crops = []
        boxes_coords = []
        for box in filtered_boxes:
            box = box.astype(int)
            # Add minimum size check
            if (box[2] - box[0]) > 20 and (box[3] - box[1]) > 20:
                crop = image.crop((box[0], box[1], box[2], box[3]))
                if crop.mode != 'RGB':
                    crop = crop.convert('RGB')
                crops.append(crop)
                boxes_coords.append(box)
        
        return crops, boxes_coords

def draw_boxes(image, boxes):
    import cv2
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)
    for box in boxes:
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return Image.fromarray(image_np)

def get_ensemble_prediction(predictions, classifier, confidence_threshold=0.7):
    model_predictions = {}
    for model_name in ['resnet', 'efficientnet', 'vgg16']:
        probs, indices = torch.topk(predictions[model_name]['probabilities'], 1)
        confidence = float(probs[0]) * 100
        predicted_class = classifier.classes[indices[0]]
        model_predictions[model_name] = {
            'class': predicted_class,
            'confidence': confidence
        }
    
    vote_count = {}
    high_confidence_votes = {}
    
    for model_name, pred in model_predictions.items():
        pred_class = pred['class']
        confidence = pred['confidence']
        vote_count[pred_class] = vote_count.get(pred_class, 0) + 1
        if confidence >= confidence_threshold:
            high_confidence_votes[pred_class] = high_confidence_votes.get(pred_class, 0) + 1
    
    prediction_info = {
        'model_predictions': model_predictions,
        'total_votes': vote_count,
        'high_confidence_votes': high_confidence_votes,
        'method_used': ''
    }
    
    high_conf_winners = [k for k, v in high_confidence_votes.items() if v == 3]
    if high_conf_winners:
        prediction_info['method_used'] = 'unanimous_high_confidence'
        return high_conf_winners[0], prediction_info
    
    high_conf_winners = [k for k, v in high_confidence_votes.items() if v >= 2]
    if high_conf_winners:
        prediction_info['method_used'] = 'majority_high_confidence'
        return high_conf_winners[0], prediction_info
    
    max_votes = max(vote_count.values())
    winners = [k for k, v in vote_count.items() if v == max_votes]
    
    if len(winners) == 1:
        prediction_info['method_used'] = 'simple_majority'
        return winners[0], prediction_info
    
    max_confidence = 0
    best_prediction = None
    
    for pred_class in winners:
        for model_pred in model_predictions.values():
            if model_pred['class'] == pred_class and model_pred['confidence'] > max_confidence:
                max_confidence = model_pred['confidence']
                best_prediction = pred_class
    
    prediction_info['method_used'] = 'highest_confidence'
    return best_prediction, prediction_info

def display_product_info(product_name):
    if product_name.lower() in PRODUCT_INFO:
        info = PRODUCT_INFO[product_name.lower()]
        st.markdown(f"""
            <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h3 style='color: #1e88e5;'>Product Information: {product_name.title()}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        sections = [
            ('🔢 Calories', 'calories'),
            ('🥗 Nutritional Value', 'nutrients'),
            ('👅 Taste Profile', 'taste'),
            ('❤️ Health Benefits', 'benefits')
        ]
        
        for title, key in sections:
            with st.expander(title):
                st.markdown(info[key])

def main():
    st.set_page_config(
        page_title="Agricultural Product Classifier & Nutrition Guide",
        page_icon="🌱",
        layout="wide"
    )

    # Ajout de la barre de navigation
    st.sidebar.title("Navigation")
    
    # Bouton pour retourner à la page d'accueil
    if st.sidebar.button("🏠 Return to Home"):
        home_path = get_absolute_path("home.py")
        if os.path.exists(home_path):
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(home_path))
            subprocess.Popen(["streamlit", "run", home_path])
            os.chdir(current_dir)
        else:
            st.sidebar.error(f"Path not found: {home_path}")
    
    # Bouton pour aller à l'application YOLO
    if st.sidebar.button("🔍 Object Detection (YOLO)"):
        yolo_path = get_absolute_path("Object-Detection-Yolo/yolo_app.py")
        if os.path.exists(yolo_path):
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(yolo_path))
            subprocess.Popen(["streamlit", "run", yolo_path])
            os.chdir(current_dir)
        else:
            st.sidebar.error(f"Path not found: {yolo_path}")
    
    # Séparateur dans la barre latérale
    st.sidebar.markdown("---")

    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button { 
            width: 100%; 
            background-color: #1e88e5; 
            color: white;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #1e88e5;
            margin: 1rem 0;
            background-color: #f8f9fa;
        }
        .model-header-resnet {
            background-color: #1e88e5;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .model-header-efficientnet {
            background-color: #43a047;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .model-header-vgg16 {
            background-color: #e53935;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .confidence-bar-container {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            margin: 5px 0;
            overflow: hidden;
        }
        .confidence-bar-fill-resnet {
            height: 100%;
            background-color: #1e88e5;
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
        }
        .confidence-bar-fill-efficientnet {
            height: 100%;
            background-color: #43a047;
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
        }
        .confidence-bar-fill-vgg16 {
            height: 100%;
            background-color: #e53935;
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌱 Agricultural Product Classifier & Nutrition Guide")
    st.markdown("""
    This application helps you:
    1. Detect and segment agricultural products in images
    2. Classify each product using three advanced AI models
    3. Learn about the nutritional value and health benefits of each product
    """)

    try:
        with st.spinner("Loading models... Please wait."):
            resnet_path = os.path.join(Config.MODEL_SAVE_DIR, 'model_acc_0.952_epoch_15.pth')
            efficientnet_path = os.path.join(Config.MODEL_SAVE_DIR, 'efficientnet_model_acc_97.151_epoch_19.pth')
            vgg16_path = os.path.join(Config.MODEL_SAVE_DIR, 'vgg16_model_acc_96.581_epoch_18.pth')
            
            classifier = TripleModelPredictionPipeline(resnet_path, efficientnet_path, vgg16_path)
            segmentation_model = SegmentationModel()
            st.success("✅ Models loaded successfully!")

        st.header("📸 Image Input")
        input_method = st.radio("Choose input method:", ["Upload Image", "Take Photo"])
        
        image_file = None
        if input_method == "Upload Image":
            image_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png'])
        else:
            image_file = st.camera_input("Take a photo:")

        if image_file is not None:
            image = Image.open(image_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.header("Original Image")
                st.image(image, use_column_width=True)

            with st.spinner("Detecting objects..."):
                crops, boxes = segmentation_model.segment_image(image)

            with col2:
                image_with_boxes = draw_boxes(image, boxes)
                st.header("Detected Objects")
                st.image(image_with_boxes, use_column_width=True)

            if len(crops) == 0:
                st.warning("No objects detected in the image. Try uploading a clearer image.")
            else:
                st.success(f"✅ Detected {len(crops)} object(s)")

                for idx, crop in enumerate(crops):
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <h2>Object {idx + 1}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([1, 1, 2])

                    with col1:
                        st.image(crop, caption=f"Detected Object {idx + 1}")

                    with col2:
                        predictions = classifier.predict_image(crop)
                        model_colors = {
                            'resnet': ('model-header-resnet', 'confidence-bar-fill-resnet'),
                            'efficientnet': ('model-header-efficientnet', 'confidence-bar-fill-efficientnet'),
                            'vgg16': ('model-header-vgg16', 'confidence-bar-fill-vgg16')
                        }

                        for model_name in ['resnet', 'efficientnet', 'vgg16']:
                            header_class, fill_class = model_colors[model_name]
                            st.markdown(f"""
                            <div class='{header_class}'>
                                {model_name.upper()} Predictions
                            </div>
                            """, unsafe_allow_html=True)

                            pred = predictions[model_name]
                            probs, indices = torch.topk(pred['probabilities'], 3)

                            for prob, class_idx in zip(probs, indices):
                                confidence = float(prob) * 100
                                st.markdown(f"""
                                    {classifier.classes[class_idx]}: {confidence:.1f}%
                                    <div class="confidence-bar-container">
                                        <div class="{fill_class}"
                                             style="width: {confidence}%;">
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    with col3:
                        best_prediction, prediction_info = get_ensemble_prediction(predictions, classifier)

                        st.markdown("### Final Prediction")
                        method_descriptions = {
                            'unanimous_high_confidence': f'✨ All models confidently predict this is a {best_prediction}!',
                            'majority_high_confidence': f'👍 Majority of models agree this is a {best_prediction}',
                            'simple_majority': f'📊 Based on majority voting, this appears to be a {best_prediction}',
                            'highest_confidence': f'🎯 Highest confidence prediction indicates this is a {best_prediction}'
                        }
                        st.markdown(f"""
                            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                                <h4 style='color: #1e88e5;'>{method_descriptions[prediction_info['method_used']]}</h4>
                            </div>
                        """, unsafe_allow_html=True)

                        with st.expander("View Model Predictions"):
                            for model_name, pred in prediction_info['model_predictions'].items():
                                st.write(f"{model_name.upper()}: {pred['class']} ({pred['confidence']:.1f}%)")

                        display_product_info(best_prediction)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check if all model files and paths are correct.")

if __name__ == "__main__":
    main()