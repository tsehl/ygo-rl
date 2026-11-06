// Lecture seule des métadonnées des choix de l'oracle épinglé. Aucune règle ni script modifié.
#include <iterator>
#include "duel.h"
#include "field.h"
#include "effect.h"
#include "card.h"
extern "C" int YGO_EffectEvent(void* raw, unsigned index, unsigned* code, unsigned* event) {
    if(!raw) return 0;
    const auto& choices = static_cast<duel*>(raw)->game_field->core.select_chains;
    if(index >= choices.size()) return 0;
    auto it = choices.begin();
    std::advance(it, index);
    if(!it->triggering_effect || !it->triggering_effect->handler) return 0;
    *code = it->triggering_effect->handler->data.code;
    *event = it->triggering_effect->code;
    return 1;
}
