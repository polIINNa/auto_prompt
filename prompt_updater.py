from __future__ import annotations

import os
import json
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.gigachat import GigaChat

load_dotenv()

gigachat = GigaChat(credentials=os.getenv('GIGA_CREDENTIALS'), scope=os.getenv('GIGA_SCOPE'),
                    model=os.getenv('GIGA_MODEL'), temperature=0.00000001)

one_point_prompt_tmpl = """Тебе будет дан промпт для большой языковой модели. Также тебе будет дан критерий для оценки промпта.
Твоя задача - определить, подходит ли данный промпт под критерий.
В ответ выведи одно слово: "да" - если промпт подходит под критерий, "нет" - если промпт не подходит под критерий.
--------------------
Промпт: {manager_prompt}

Критерий: {point}
--------------------
Теперь реши поставленную задачу. Важно строго соблюдать формат вывода.
"""

upd_by_recs_prompt_tmpl = """
Тебе будет дан промпт для большой языковой модели. Твоя задача - улучшить данный промпт. Ниже приведены критерии для улучшения промпта.

--------------------

Новый улучшенный промпт должен соответствовать следующим критериям:
{missing_points}

--------------------

Исходный промпт: {manager_prompt}

Улучшенный промпт: 

"""

add_few_shot = """
Тебе будет дан промпт, и твоя задача - добавить в него 3 few-shot примера, при этом сохранив весь текст исходного промпта.

-------
Исходный промпт: {zero_shot_prompt}
-------
Пожалуйста, добавь 3 few-shot примера в конце исходного промпта. 
Few-shot пример должен содержать входные данные и выходные данные, которые описываются в промпте.
Обязательно оставь и верни текст исходного промпта без изменений и в полном объеме. Не убирай, не меняй и не сокращай ничего в исходном промпте, иначе будут негативные последствия.

Исходный промпт с few-shot примерами:
"""

POINTS = [
    "Наличие роли, экспертности или знаний в определенной области. В промпте должна быть указана роль модели, ее экспертность или базовые знания в конкретной области (например, 'Ты - специалист по финансовым услугам').",
    "Наличие инструкции. В промпте должна быть четко указана задача, которую должна выполнить языковая модель.",
    "Наличие контекста. В промпте должна быть предоставлена дополнительная информация, которая окружает основную задачу или вопрос и помогает лучше понять суть задачи или вопроса (например тема, цель вопроса и так далее).",
    "Наличие описания выходных данных. В промпте должно быть четко указано, что должно быть получено в результате выполнения задачи, которая описывается в промпте. Также должен быть указан формат выходных данных.",
    "Ясность и конкретность. Промпт должен быть чектим, понятным, детализированным и описательным.",
    "Однозначность инструкций в промпте, отсутствие двусмысленности",
    "Утвердительные операции. Формулируйте задачу для языковой модели так, чтобы она четко указывала на действия, которые необходимо выполнить, избегая указания на то, что делать не следует.",
    "Наличие few-shot примеров."
]


def get_upd_prompt_by_recs(manager_prompt: str) -> str:
    upd_by_recs_chain = PromptTemplate.from_template(template=upd_by_recs_prompt_tmpl) | gigachat
    add_few_shot_chain = PromptTemplate.from_template(template=add_few_shot) | gigachat
    missing_points = _get_missing_points(manager_prompt=manager_prompt, points=POINTS)
    if missing_points == []:
        return 'Промпт хороший, в серьезных изменениях не нуждается.'
    else:
        if 'Наличие few-shot примеров.' in missing_points:
            missing_points.remove('Наличие few-shot примеров.')
            if missing_points == []:
                upd_few_shot_prompt = add_few_shot_chain.invoke({'zero_shot_prompt': manager_prompt}).content
                return upd_few_shot_prompt
            else:
                upd_zero_shot_prompt = upd_by_recs_chain.invoke(
                    {'manager_prompt': manager_prompt, 'missing_points': '\n'.join(missing_points)}).content
                upd_few_shot_prompt = add_few_shot_chain.invoke({'zero_shot_prompt': upd_zero_shot_prompt}).content
                return upd_few_shot_prompt
        else:
            return upd_by_recs_chain.invoke(
                {'manager_prompt': manager_prompt, 'missing_points': '\n'.join(missing_points)}).content


def _get_missing_points(manager_prompt: str, points: List[str]) -> List[str]:
    one_point_prompt = PromptTemplate.from_template(template=one_point_prompt_tmpl)
    one_point_chain = one_point_prompt | gigachat
    missing_points = []
    for point in points:
        res = one_point_chain.invoke({'manager_prompt': manager_prompt, 'point': point}).content
        if res == 'нет':
            missing_points.append(point)
    return missing_points

