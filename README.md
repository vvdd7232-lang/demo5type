# Demo 5 Type — прототип гуманоидной симуляции

Это **первый исполняемый каркас**, а не готовый Isaac Lab-проект и не обученная модель. В репозитории есть независимое от симулятора ядро поведения, сценарий для проверки переходов и черновой hook для интерактивной цели ProtoMotions. Isaac Sim, GPU, motion-файлы и веса сюда не включены; обучение на этой машине не запускалось.

## Что уже работает

- `demo5type.behavior` — конечный автомат `explore → interest → fright`: в `explore` предлагает медленное движение (0,15 м/с по умолчанию) и сканирование взглядом, в `interest` выбирает ближайший интерактивный объект с гистерезисом, а в `fright` выдаёт сигнал защитной позы и направления отступления. Люди исключены из целей захвата. Это намерения для адаптера, не команды мышцам.
- `demo5type.control` — переводит намерения в режимы `wander/approach/reach/retreat/hold`, ограничивает скорости и задаёт точку подхода с безопасным отступом от центра объекта.
- `demo5type.observations` — фиксированный вектор из смещений «голова → объект», «кисть → объект» и флага видимости.
- `demo5type.rewards` — эталонные скалярные награды за взгляд, дотягивание, контакт пальцев и подъём. Это Python-формулы, не готовые GPU-ядра для RL.
- `demo5type.domain_randomization` — воспроизводимые выборки трения пола и импульсов. Они **не применяются** к физике, пока не добавлен backend-адаптер.
- `demo5type.preflight` — диагностическая проверка Python, PyTorch/CUDA и установленных пакетов Isaac.
- `integrations/protomotions/reach_inference.py` — inference-hook для ручного native reaching.
- `integrations/protomotions/runtime_adapter.py` — dispatcher для ProtoMotions `set_location` / `set_reaching` / `set_steering` с защитой от переустановки цели на каждом physics step. Хват и защитная поза требуют явно переданного auxiliary-handler.
- `integrations/protomotions/behavior_inference.py` — script-driven конфигурация native goals для собственного inference loop.

`approach_target_world` указывает на объект; `demo5type.control` переводит его в точку остановки с отступом. Это геометрический standoff без проверки препятствий: backend всё ещё должен строить collision-free траекторию. Выходы автомата нельзя напрямую трактовать как физические усилия.

## Локальный запуск без Isaac Sim

Нужно только Python 3.10+; сторонние зависимости для ядра не требуются.

```bash
python -m unittest discover -s tests -v
python -m demo5type.demo
python -m demo5type.preflight
```

`demo` — сухой сценарий сенсорных входов. Он показывает режимы автомата и сформированные `ControlCommand`, но не рисует скелет и не симулирует мышцы, столкновения или хват.

Диагностику можно получить в JSON; строгие флаги пригодятся уже на машине с NVIDIA GPU:

```bash
python -m demo5type.preflight --json
python -m demo5type.preflight --require-cuda --require-sim
```

## Windows EXE

`demo5type.exe` — **консольное dry-run демо** FSM, не Isaac Sim, не обученная модель и не графическое приложение. Оно не включает PyTorch, Isaac Lab или GPU backend.

На Windows x64 с Python 3.11 соберите one-file EXE:

```powershell
.\scripts\build_windows.ps1
```

Результат: `dist\demo5type.exe`; рядом создаётся SHA-256 файл. Скрипт установит PyInstaller в локальное `.venv-build`. Тот же бинарник собирается workflow **Build Windows executable**; его artifact называется `demo5type-windows-x64`.

EXE не подписан цифровым сертификатом, поэтому Windows может показать предупреждение SmartScreen. Не отключайте защиту системы ради запуска; сверяйте SHA-256 и собирайте из исходников, если Windows блокирует неизвестный бинарник.

## Linux / Arch Linux x86_64

Сборка выполняется внутри настоящего Arch Linux x86_64 контейнера workflow **Build Arch Linux binary**; artifact: `demo5type-arch-linux-x86_64`. Artifact загружается как ZIP; после распаковки ZIP распакуйте вложенный tarball, чтобы сохранить право исполнения:

```bash
tar -xzf demo5type-arch-linux-x86_64.tar.gz
sha256sum -c demo5type.sha256
./demo5type
```

Если у вас уже скачана прежняя версия artifact напрямую с бинарником внутри ZIP, после распаковки выполните `chmod +x demo5type`, затем `./demo5type`.

Для сборки непосредственно на Arch запустите:

```bash
bash scripts/build_linux.sh
```

Скрипт создаёт изолированное `.venv-build-linux`, устанавливает PyInstaller и smoke-тестирует файл. Результат: `artifacts/linux-arch-x86_64/demo5type` и SHA-256. Бинарник не требует системного Python. Это всё ещё консольное dry-run демо, не Isaac Sim и не GPU-симуляция.

## Подключение к симулятору

### Isaac Lab: сначала короткий smoke-тест

Инструкции установки нужно брать из официального [Isaac Lab](https://github.com/isaac-sim/IsaacLab) и проверять совместимость с конкретной версией Isaac Sim, драйвера и видеокарты. В текущем upstream Isaac Lab AMP-среда ходьбы зарегистрирована как `IsaacContrib-Humanoid-AMP-Walk-Direct`; это не обещание, что конкретная GPU запустит 4096 сред.

Из корня установленного checkout Isaac Lab начните с малого количества сред:

```bash
uv run isaaclab random_agent \
  --task IsaacContrib-Humanoid-AMP-Walk-Direct \
  --num_envs 8
```

После успешного reset/step можно отдельно проверить короткий запуск обучения через `skrl`:

```bash
uv run --extra skrl isaaclab train \
  --rl_library skrl \
  --task IsaacContrib-Humanoid-AMP-Walk-Direct \
  --num_envs 64 \
  --max_iterations 10
```

Это лишь smoke-тест встроенной AMP-среды Isaac Lab; он **не обучает** логику этого репозитория. Увеличивать число сред следует по результатам измерений памяти и скорости.

### ProtoMotions: интерактивная цель руки

Пример `integrations/protomotions/reach_inference.py` использует штатную задачу `reaching` ProtoMotions и имя правой кисти `R_Hand` для стандартного скелета SMPL. Он требует **уже обученный совместимый MaskedMimic-чекпойнт**, файл движений и установленный ProtoMotions; чекпойнты и motion data не поставляются. На другом роботе измените имя тела и проверьте, что оно входит в conditionable body set чекпойнта.

Пример команды из корня совместимого ProtoMotions checkout:

```bash
python protomotions/inference_agent.py \
  --simulator isaaclab \
  --num-envs 1 \
  --checkpoint /path/to/native-goal-student.ckpt \
  --motion-file /path/to/compatible-motion.pt \
  --inference-config /path/to/demo5type/integrations/protomotions/reach_inference.py
```

Это даёт **цель для кисти**, а не физический хват банки. Чтобы получить хват, нужно добавить предмет в сцену, коллизии пальцев, проверку контакта и физическое ограничение/модель захвата. Hook не запускает `BehaviorStateMachine`: `demo5type.control` переводит её результат в backend-neutral команды, а `runtime_adapter.py` умеет передать их штатному `env.control_manager.components["masked_mimic"]` из собственного inference loop.

Для такого loop используйте `integrations/protomotions/behavior_inference.py` (без клавиатурных binding) и `ProtoMotionsCommandDispatcher`:

```python
from integrations.protomotions.runtime_adapter import ProtoMotionsCommandDispatcher

dispatcher = ProtoMotionsCommandDispatcher()  # один экземпляр на однородный batch
control = env.control_manager.components["masked_mimic"]
dispatcher.update(
    control, env_ids, command,
    now_s=simulation_time_s,
    base_position_world=base_position,
    on_auxiliary_intent=handle_guard_and_grasp,
)
```

`update(...)` вызывается после получения сенсоров и расчёта команды; dispatcher ограничивает частоту обновления. Режимы `wander/retreat` используют ProtoMotions `set_steering` и требуют чекпойнт с velocity-goal каналом. `protect_head` и `grasp_requested` намеренно не игнорируются: без вашего simulator-specific auxiliary-handler dispatcher остановится с `NotImplementedError`. Самого runtime loop, perception-модуля, физического предмета и обработчика хвата в репозитории пока нет.

## Важная поправка к исходному плану установки

Состояние upstream проверено **11 ноября 2026 года**. Актуальная документация ProtoMotions 3 описывает Isaac Lab 3 / Isaac Sim 6 и Python 3.12 на Linux x86_64. Поэтому сочетание Python 3.10/3.11, произвольного CUDA-wheel PyTorch и старого Windows-командного файла `isaaclab.bat --install` может не соответствовать выбранному стеку ProtoMotions. Следуйте pinned-инструкциям именно той версии, которую устанавливаете; не переустанавливайте драйверы и CUDA «наугад». Само чистое ядро этого репозитория остаётся совместимым с Python 3.10+.

Начинайте с одной видимой среды или небольшого headless smoke-теста. «4096» — число параллельных копий среды в конфигурации, а не гарантия производительности или обещание, что 4000 персонажей поместятся в память любой видеокарты. Результат зависит от GPU, версии физического backend, рендера, сенсоров и разрешения сцены.

Motion capture тоже требует проверки условий доступа: не считайте AMASS, CMU или найденные `.bvh`/`.fbx` автоматически свободными для скачивания, обучения или распространения. Используйте официальные источники, лицензии и правила датасета; большие motion-файлы и веса не коммитьте в этот репозиторий.

## Следующие технические шаги

1. Написать Isaac Lab/ProtoMotions-адаптер наблюдений: позиции головы/кисти/объекта, локальные координаты и пакетная обработка нескольких сред.
2. Добавить custom ProtoMotions inference loop: считывать сенсоры, вызывать FSM и `ProtoMotionsCommandDispatcher`, затем проверить `approach/reach/retreat` на одной среде. Готовый runner/перцепция пока не включены.
3. Добавить физический объект и безопасную модель захвата; обучать награду только после проверки контактов и сброса среды.
4. Перенести эталонные награды в поддерживаемые выбранным RL-стеком tensor kernels и применять рандомизацию через API конкретного backend.
5. Проверить стабильность на малом числе сред, затем постепенно повышать параллелизм.

Любая интеграция с реальным роботом требует отдельной аппаратной защиты и не входит в этот прототип.

## Официальные ссылки

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab: reinforcement learning](https://github.com/isaac-sim/IsaacLab/blob/main/docs/source/concepts/reinforcement_learning.rst)
- [Isaac Lab: текущая AMP humanoid-среда](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- [ProtoMotions: установка](https://nvlabs.github.io/ProtoMotions/getting_started/installation.html)
- [ProtoMotions: быстрый старт](https://nvlabs.github.io/ProtoMotions/getting_started/quickstart.html)
- [ProtoMotions: native MaskedMimic goals](https://nvlabs.github.io/ProtoMotions/user_guide/native_goals.html)
