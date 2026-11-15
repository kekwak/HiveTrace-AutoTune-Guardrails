# Baseline: закреплённые режимы

## Schema 4: исполнение авторов HF/SDK, 12 ноября

Источники истины для запуска: [полные пресеты](../configs/v4) и закреплённый
[RTX-lock](../env/team_rtx.lock.json). Baseline следует авторскому HF/SDK-пути
в этой среде, а не настройкам командного профилирования. Это не побитовое
воспроизведение другой версии Transformers из примера в карточке.

| Модель | Precision baseline | Attention и реализация |
| --- | --- | --- |
| [HaloGuard](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B/blob/85ff68dd063a7e58487a4251824fd36a20977d5c/README.md) | BF16, значение `auto` для закреплённого BF16 checkpoint | Нативный выбор full attention в Transformers; linear attention без подмены функций измерителем |
| [GLiNER](https://huggingface.co/hivetrace/gliner-guard-omni/blob/763f8aa9a7714dd2c3264f3751acdd5d854c5ac4/README.md) | FP32 | Штатный SDK, standard encoder; без compile/ORT/quantize |
| [Koala](https://huggingface.co/KoalaAI/Text-Moderation/blob/838ade0edb66dcffc5532d08ff6ed5c899abfb5c/README.md) | FP32 | Штатный HF-loader, без override attention |
| [Llama Prompt Guard](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M/blob/11614a155199674a0a95e6602d6ab0417b790ed0/README.md) | FP32 | Штатный HF-loader; ранее согласованное зеркало 22M сохранено |
| [Laya](https://huggingface.co/convaiinnovations/laya/blob/7b928d828b7b0e022f929d9bd2e44165aa270148/README.md) | FP32-веса и native SDK autocast | SDK SDPA; измеритель больше не принуждает efficient kernel; `fast=False` как в SDK по умолчанию |
| [HiveTraceGuard-Pro](https://huggingface.co/hivetrace/HiveTraceGuard-Pro/blob/ed008276c96253e14885e6ade3b7b6d58daa2480/README.md) | BF16 | Штатный HF-loader, без принудительного SDPA math |
| [OpenGuardrails 0124](https://huggingface.co/openguardrails/OpenGuardrails-Text-4B-0124/blob/9bdd1fb549e01b5f6aae4e9c395998492a08830b/README.md) | BF16, значение `auto` для закреплённого checkpoint | Штатный HF-loader, без принудительного SDPA math; зеркало 0124 и task2510 сохранены |
| [ShieldGemma](https://huggingface.co/google/shieldgemma-2b/blob/d1dffc9c8c9237a90aab09c61383791e718ef9e8/README.md) | BF16 | Eager сохраняет авторский attention soft-capping; исключение для закреплённого Transformers описано ниже |

`attention="native"` означает отсутствие `attn_implementation` в HF-loader,
а `sdpa_backend=null` не ограничивает штатный выбор kernel. Это **не** обещание
одного kernel для всех форм или GPU. Фактический attention сохраняется после
загрузки, операторы и kernels проверяются отдельным preflight trace. Явные
`eager`, `sdpa` и ограничения math/efficient остаются для отдельных экспериментов;
ошибка исполнения не вызывает повторный запуск с другой реализацией.

**Shield:** в Transformers 5.20.1 SDPA принимает, но не применяет `softcap`.
Поэтому выбран eager, чтобы сохранить вычисления модели, а не воспроизводить
[известный дефект HF-default](https://github.com/huggingface/transformers/issues/49810).
Это явное ограничение текущего стека, не универсальное требование ко всем моделям.

**Halo:** штатный Transformers может уже использовать установленный FLA.
`halo_fla.json` теперь сравнивает этот же авторский baseline с явным linear-only
FLA/reference-conv вариантом в BF16; он не является прежним FP16 reference→FLA
экспериментом и не обещает ускорения. Внешние незакреплённые Hub kernels по-прежнему
отключены (`USE_HUB_KERNELS=NO`); доступные в lock пакеты выбираются самим Transformers.

Сохраняются GPU 0 без offload, полный согласованный scores-путь, роли, шаблоны,
пороги, tie/stop rules и лимиты. В частности, Open не возвращается к непригодному
native0124-промпту. Обновление precision/backend не меняет preparation identity
или опубликованный корпус. Оно меняет конфигурацию исполнения: прежние reference,
approvals и smoke не подтверждают новые пресеты.
[Новый Open-smoke](../results/u1_handoff/2026-11-12-v4-native-open/README.md)
прошёл на RTX 5090: BF16, SDPA efficient, B1x64/B2x32/B3x16/B4x8.
Остальные новые режимы этим не проверены; проверка качества и новых скоростных
серий не выполнялась.

## История: schema 3

> Исторический снимок schema 3. Для schema 4 источником истины являются полные
> [восемь конфигов](../configs/v4), а не каталог по имени checkpoint.
> [Новый порядок baseline/search/confirm](u1_handoff.md#4-baseline--поиск--подтверждение).

Актуально на **8 ноября 2026**, формат конфигураций 3.
[Запуск](u1_handoff.md) · [полный протокол](measurement_protocol_v1.md) ·
[готовность](u1_handoff_readiness.md).

## Текущий контроль

Контроль скорости и изменений вердиктов — PyTorch eager / штатный SDK в указанной
ниже точности. В него не добавляются compile, ORT, квантование, CUDA Graph capture,
vLLM или межзапросное кеширование. Eager не означает принудительную замену каждого
штатного efficient SDPA: фактический dispatch сохраняется в контракте и доказательствах.

| Модель / ключ | Baseline | Решение и backend | Обязательные бакеты |
| --- | --- | --- | --- |
| HaloGuard / `halo` | FP16 | Авторский prompt, thinking off, `Safety: `; unsafe ≥0.65; reference linear/conv/norm и eager full attention | B1–B4 |
| GLiNER / `gliner` | FP32 | Полный extract, схема safety; argmax, tie→safe; штатный encoder | B1 |
| Koala / `koala` | FP32 | Argmax девяти native-классов; tie→меньший индекс | B1 |
| Llama Prompt Guard / `llama` | FP32 | Benign/malicious argmax, tie→меньший индекс; явно выбранное зеркало hipocap | B1 |
| Laya / `laya` | FP32-веса + нативный autocast SDK | SDK predict, danger choice A/B, исходная температура/голова и efficient SDPA | B1 |
| HiveTraceGuard-Pro / `hive` | FP16 | Собственный шаблон, greedy generate до одного токена, native EOS, SDPA math | B1–B4 |
| OpenGuardrails / `open` | FP16 | Веса/tokenizer 0124 + task-шаблон 2510, greedy до 10 токенов, native EOS, SDPA math; verdict и категория | B1–B4 |
| ShieldGemma / `shield` | FP32 | Dangerous Content для запроса/ответа; eager attention с softcap, Yes/No softmax, tie→No | B1–B4 |

Все checkpoint/tokenizer/template revisions закреплены в
[реестре](../speed_dataset/contracts.json), правила и хеш подготовки —
в [контракте](../u1_benchmark/contracts.py). [Матрица ролей](u1_handoff.md#входы-моделей)
определяет оцениваемое сообщение. Не используется общий неявный safe/unsafe mapping.

**Laya, 7 ноября:** адаптер сохраняет выбор закреплённого SDK без изменения config
или FP32-весов. В текущем checkpoint задано `amp_dtype=bf16`: SDK на T4 выбирает
FP16, а на поддерживаемых новых GPU использует BF16. Это описание поведения SDK,
не таблица выбора dtype внутри измерителя. Фактический режим записывается в
`metadata.precision`; CPU fallback, отключение AMP и последующая смена режима
останавливают worker. Нативный baseline не обязан иметь одинаковую precision на
разном железе; такие конфигурации не объединяются. Патч прошёл
[smoke на RTX 4090](../results/u1_handoff/2026-11-08-laya-rtx4090/validation.json)
с FP32-весами и BF16 autocast; RTX 5090 не проверена.

В инженерном сравнении baseline и candidate используют одинаковые dtype параметров
и autocast; в параметрическом или квантованном отличие фиксируется явно. Все workers
одного варианта, включая memory и quality, сохраняют один паспорт precision.
Исторические отчёты без паспорта остаются читаемыми, но не подтверждают новую политику.

Будущие замеры: RTX 4090 либо RTX 5090, один lock, GPU 0, одинаковые входы и порядок
внутри пары. Оборудование и группы совместимости разделяются. Baseline измеряется
заново рядом с candidate; чужой старый CSV не подставляется в знаменатель.
FP32 не является обязательным скрытым контролем всех FP16-пресетов: отдельная
численная FP32-регрессия должна быть явно выполнена и подписана, если требуется.

Корпус общий, но модельные бакеты независимы, по полной нативной длине.
B1 1–512, B2 513–1024, B3 1025–2048, B4 2049–4096; по 400 уникальных эффективных
входов на обязательный бакет. Пересечения длин всех моделей и старые фиксированные
квоты Wikipedia/синтетика не используются. Кандидат получает списки baseline без переотбора.

**Текущий статус:** восемь baseline прошли [T4-smoke 4 ноября](../results/u1_handoff/2026-11-04-review-smoke/validation.json).
Отдельно подтверждён только Laya baseline-smoke на RTX 4090. Остальные модели на
RTX, RTX 5090, полная новая baseline-таблица и quality-проверка не подтверждены.
Для Open обязательна CPU-переиндексация по инструкции; родной шаблон 0124 —
исторический диагностический профиль, не текущий контроль.

## Историческая спецификация v3/T4, 30 октября

Ниже сохранено прежнее обоснование и источники. Его пересечения бакетов, T4,
условия доступа и ещё не выполненные проверки описывают состояние того этапа;
они не переопределяют текущие режимы выше.

Рабочий выбор У1 от 30 октября 2026 по поручению пользователя. Это спецификация следующего измерителя, не выполненные новые замеры и не утверждение, что все участники согласовали эти режимы. [Протокол](measurement_protocol_v1.md). Удалённые ветки повторно проверены 30 октября; новых коммитов относительно перечисленных источников не обнаружено.

## 1. Основной baseline

Контроль скорости и вердиктов — PyTorch eager / штатный SDK в выбранной точности. Без добавленных compile, ORT, vLLM, CUDA Graph capture, квантования, межзапросного prefix caching или наших kernel patches. Eager означает отсутствие компиляции графа, а не принудительную замену штатного efficient SDPA на медленный attention. Attention dispatch фиксируется отдельно; обновление библиотеки не должно молча менять его.

Одинаковые T4, общий lock, входы и контракт внутри baseline/candidate-пары обязательны. Старые CSV показывают происхождение выбора, но контроль заново измеряется общим скриптом в той же VM. FP32 остаётся дополнительным численным reference там, где доступен; не используем более медленный знаменатель ради большого speedup. Для FP16 baseline маркер изменений вердиктов также считается относительно FP16 baseline, а сравнение с FP32 публикуется отдельно.

| Модель | Выбранный baseline на T4 | Контракт и внимание | Источник |
| --- | --- | --- | --- |
| HaloGuard 0.8B | FP16 eager | Авторский prompt, thinking off, один forward, safe/unsafe scores, порог 0.65; full attention eager, linear/conv/norm reference | [Наше подтверждение](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/tree/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla), [адаптер](../experiments/adapters.py) |
| GLiNER Guard Omni | FP32 eager | Полный extract, схема safety, argmax, tie → safe; standard encoder attention. Не encoder-only | [Адаптер](../experiments/adapters.py) |
| Koala Text-Moderation | FP32 eager | Штатный DeBERTa, argmax исходных 9 классов; не заменяем бинарным порогом | [Notebook](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/c94afd857838fed7be3e1505734cd3a5f89ebdcf/notebooks/koala_profile.ipynb) |
| Llama Prompt Guard 2 22M | FP32 eager | DeBERTa, argmax benign/malicious. В историческом профиле только forward; общий скрипт добавит подготовку и получение вердикта | [Исходник](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/2f65d6fe30f29f2d36f8ee397db6380b81e1a2d7/src/profiling/profile.py) |
| Laya English, корневой checkpoint | FP32-веса + штатный autocast FP16 на запрос, inference_mode | SDK predict, один choice-вопрос с двумя вариантами, исходные температура/голова и efficient SDPA. Persistent autocast и FP16-веса — кандидаты | [Отчёт](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/2eefe392bfac35099b63340a3289425738d692eb/results/profiling/laya/README.md) |
| HiveTraceGuard-Pro | FP16 eager execution | Greedy generate одного токена, штатный шаблон и SDPA math из измеренного baseline; без repeat-KV patch | [Исходник](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/2e9d79efb091610c775b569f5a8d9dff1e706c1e/src/profiling/profile.py), [dispatch](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/2e9d79efb091610c775b569f5a8d9dff1e706c1e/results/profiling/HiveTraceGuard-Pro/README.md) |
| OpenGuardrails Text 4B 0124 | FP16 eager execution | Greedy generate до 10 новых токенов, исходные stop rules, включая категорию в затратах; SDPA math. Шаблон 2510 из рабочего notebook фиксируется явно | [Notebook](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/709ec61c39ea93912fab252eaf9de4671fa5de04/notebooks/openguardrails_profile.ipynb), [обоснование шаблона](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/709ec61c39ea93912fab252eaf9de4671fa5de04/results/profiling/OpenGuardrails-Text-4B-0124/README.md) |
| ShieldGemma 2B | FP32 eager | Eager attention с softcap; Dangerous Content, один forward, Yes/No scores. Для общего скрипта выбран argmax (tie → No); не выдаём это за неизвестный авторский порог | [Отчёт и notebook](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/2eefe392bfac35099b63340a3289425738d692eb/results/profiling/shieldgemma-2b/README.md) |
| WildGuard, резерв | Пока не назначен | Нужны доступ, checkpoint config и проверка памяти T4 | [Карточка](https://huggingface.co/allenai/wildguard) |

Не приводим всё к одному dtype: FP16 уже рабочий режим части коллег, для ShieldGemma на T4 проверен FP32, у Laya существенны и dtype весов, и autocast. Смена dtype — отдельный precision-эксперимент, не чистая настройка runtime.

Перед первым запуском адаптер фиксирует revision весов/tokenizer, полный prompt/schema, mapping классов, tie/stop rules, dispatch и source hash. Часть коллег грузила незакреплённый main; в измеритель это не переносим без сохранения точного SHA. В старом Llama-профиле используется зеркало hipocap: происхождение и доступ фиксируем отдельно, идентичность оригиналу по имени не предполагаем.

Для корпуса скорости v3 явно выбран `hipocap/Llama-Prompt-Guard-2-22M` на ревизии `05f1919555db0e711b73cbb7ec2424702b918c20`. Проверяется только tokenizer/preparation на CPU; это не проверка весов, качества или тождества историческому запуску коллеги. Будущий модельный baseline должен использовать тот же явно закреплённый источник, без подмены оригиналом по совпадению названия.

OpenGuardrails: замена generate на одну пару логитов — отдельный контракт, не незаметная часть vLLM-ускорения. При текущем контракте кандидат сохраняет правила категории и остановки; бинарный verdict, категория и длина выхода сохраняются отдельно. Ошибка парсинга/метка вне словаря не превращаются в safe. Koala сохраняет native multiclass labels, а не искусственные safe/unsafe.

Общий lock ещё не проверен на чужих моделях; Laya/ShieldGemma исторически использовали другую Transformers. Несовместимость или изменение контракта дают unsupported, а не автоматическую замену пакетов/реализации. Старые результаты не переоцениваем по новой среде.

## 2. Обязательные бакеты

Полный вход после tokenizer/template/prompt/schema и специальных токенов, без искусственного padding: B1 = 1–512, B2 = 513–1024, B3 = 1025–2048, B4 = 2049–4096. Пустой пользовательский текст тоже может иметь непустой полный вход. B4 добавлен по уточнению пользователя 30 октября; прежний трёхбакетный корпус сохранён как v1.

| Модель | Обязательный набор следующего контракта | Статус |
| --- | --- | --- |
| Laya English / GLiNER / Koala / Llama Prompt Guard | B1 | Длинные бакеты unsupported в текущем контракте; без автоматического chunking/расширения позиций |
| HaloGuard | B1, B2, B3, B4 | Новый лимит измерителя 4096 вместо исторических 1200; нужен пограничный T4 smoke |
| HiveTraceGuard-Pro / OpenGuardrails / ShieldGemma | B1, B2, B3, B4 | Контекст допускает диапазон; новый стек и весь диапазон требуют smoke |
| WildGuard | Не назначен | Не входит в покрытие до проверки |

Источники лимитов: [Laya](https://huggingface.co/convaiinnovations/laya/blob/7b928d828b7b0e022f929d9bd2e44165aa270148/README.md), [GLiNER encoder](https://huggingface.co/hivetrace/gliner-guard-omni/blob/763f8aa9a7714dd2c3264f3751acdd5d854c5ac4/encoder_config/config.json), [Koala](https://huggingface.co/KoalaAI/Text-Moderation/blob/main/config.json), [Llama](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M), [HaloGuard](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B/blob/85ff68dd063a7e58487a4251824fd36a20977d5c/config.json), [HiveTraceGuard-Pro](https://huggingface.co/hivetrace/HiveTraceGuard-Pro/blob/main/config.json), [OpenGuardrails](https://huggingface.co/openguardrails/OpenGuardrails-Text-4B-0124/blob/main/config.json); ShieldGemma — отчёт команды выше. max_position_embeddings не доказывает качество или размещение максимального контекста на T4; для DeBERTa это основание консервативного контракта, не доказательство невозможности любого более длинного forward.

Набор определяется по baseline до сравнения и одинаков для всех кандидатов. Провал baseline smoke оставляет контракт неподтверждённым, не уменьшает его задним числом. Если baseline работает, а кандидат даёт OOM/ошибку/не проходит порог хотя бы в одном обязательном бакете, кандидат не принят. Unsupported вне контракта явно перечисляются, не входят в macro и не считаются pass.

## 3. Один корпус, бакеты по пересечениям

Один версионированный корпус: ID, text, source/license, content hash, язык (если известен) и формат. Байты текстов одинаковы для всех. Для каждой модели механически строим индекс ID → user_tokens, total_tokens, bucket, supported и input hash. Это таблица для общего корпуса, не отдельный датасет переписанных текстов.

По уточнению пользователя общий состав текстов важнее заполнения всего диапазона длин. До отбора фиксируем для каждого бакета M_b — набор моделей, у которых этот бакет обязателен. B1: все восемь выбранных моделей; B2/B3/B4: HaloGuard, HiveTraceGuard-Pro, OpenGuardrails и ShieldGemma. Резервный WildGuard не включён до проверки контракта.

Текст допустим для бакета b, только если его полный вход попадает в b у каждой модели из M_b: S_b = {text: для всех m из M_b, bucket(total_tokens(m, text)) = b}. Учитываем фактическую подготовку prompt/schema и специальные токены, не только голый tokenizer.encode(text). Выбираем не менее 400 различных текстов из этого пересечения по зафиксированному seed, без просмотра скорости/вердиктов кандидатов. Получаем один список ID на бакет, одинаковый для всех участвующих в нём моделей. Внутри пары baseline/candidate также идентичны token IDs и маски; между моделями токены и их число могут различаться, несмотря на одинаковый исходный текст.

При нехватке пересечения дополняем общий исходный пул и повторяем токенизацию. Не создаём отдельные тексты для модели, не исключаем неудобную модель из M_b, не меняем шаблон, не обрезаем и не набиваем padding ради заполнения бакета. Нехватка 400 примеров или пустое пересечение — not_ready с диагностикой ограничивающих моделей. Изменение состава моделей/контракта/корпуса требует новой версии общего manifest до замеров. Ровно 1600 исходных текстов не гарантируют 400 в каждом из четырёх пересечений; повтор одного текста не восполняет покрытие.

Равномерность и покрытие от нижней до верхней границы не обязательны. Перед фиксацией версии публикуем N/min/median/p95/max и гистограммы полных длин выбранных текстов отдельно для моделей, размер исходного пула/пересечения и отказы. Вывод относится к этому набору и фактически измеренным длинам, не доказывает выигрыш на каждой возможной длине бакета. Пограничные smoke на лимитах остаются отдельными проверками выполнения, не частью обязательного равномерного распределения скорости.

Источник и доли зафиксированы в [инструкции корпуса](speed_dataset.md): Wikipedia EN/RU с закреплённой ревизией и детерминированная синтетика; на бакет 200 EN + 200 RU, по 150 корпусных и 50 синтетических на язык. Это не production-репрезентативный набор. Длинные авторские инструкции могут сильно сузить допустимые пользовательские тексты B1; фактические длины публикуются в manifest. Общие тексты улучшают сопоставимость, но не уравнивают prompt, tokenizer, dtype или качество разных моделей. Speedup по-прежнему считаем внутри модели.

При одинаковой форме dense forward длина и dtype обычно важнее языка, но полный запрос включает tokenizer, а generate может выдавать разное число токенов в зависимости от содержания. Не предполагаем строгую независимость времени от текста. Сохраняем формат, длину и фактический output_tokens; это не превращает скоростной корпус в quality-бенчмарк.

Скоростной корпус У1 и размеченный набор качества/хвостов У2 — два разных датасета с разными версиями и статусами. На первом измеряем согласованность с baseline, не accuracy/F1/FPR/FNR. Ноль flips на скорости не заменяет проверку хвостов.

## 4. vLLM как кандидат

Гипотеза: реализация инференса/адаптер vLLM даст нужное снижение p95. Это не обязательный serving и не доказанная T4-совместимость. Сначала загрузка и сохранение выбранного контракта, затем тот же парный замер. Baseline заранее на vLLM не переводится.

Batch/concurrency = 1, без незаметного prefix caching. Фиксируем dtype, длины, scores, stop rules и backend. Если нужен другой стек, контроль заново измеряется в новой совместимой среде, влияние её смены показывается отдельно; исторический CSV не заменяет контроль. Вариант с новым dtype/scoring не называется backend-only.

Для первого эксперимента предпочтителен локальный engine с той же границей text → verdict. Для HTTP-serving обе стороны получают сопоставимую обвязку, результаты отделяются от внутрипроцессных. Custom configuration не должна скрывать другую архитектуру, усечение или CPU fallback. Объём разработки адаптера и распределение моделей относятся к отдельному будущему плану.
