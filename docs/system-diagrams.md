# Архитектура и рабочие процессы OrgeMage

В данном документе описана высокоуровневая архитектура системы OrgeMage, взаимодействие компонентов и жизненный цикл выполнения запросов.

## 1. Архитектурная схема

OrgeMage выступает в роли "умного прокси", который принимает запросы по протоколу ACP (Northbound) и делегирует их выполнение специализированным агентам (Southbound).

```mermaid
flowchart TD
    subgraph Upstream ["Upstream (Клиент)"]
        UI["ACP UI / IDE Extension"]
    end

    subgraph Orchestrator ["OrgeMage (Оркестратор)"]
        NB["Northbound ACP Server"]
        
        subgraph Core ["Ядро управления"]
            Catalog["Federated Catalog
            (Реестр моделей)"]
            Engine["Orchestration Engine
            (Управление ходом выполнения)"]
            Scheduler["Task Scheduler
            (Граф выполнения и зависимости)"]
        end

        subgraph State ["Слой данных"]
            DB[(SQLite: runtime.db)]
            Snapshot["Session Snapshot
            (Состояние сессий и задач)"]
        end

        subgraph Southbound ["Southbound (Коннекторы)"]
            ConnMgr["Connector Manager"]
            
            subgraph Adapt ["Адаптеры протоколов"]
                ACP_Client["ACP Downstream Client
                (Snake/Camel Case Normalizer)"]
                Codex_Adapt["Codex App-Server Adapter
                (JSON-RPC over stdio)"]
            end
        end
    end

    subgraph Agents ["Downstream Agents (Исполнители)"]
        Gemini["Gemini CLI
        (Strict ACP)"]
        Qwen["Qwen Code
        (Standard ACP)"]
        Codex["Codex
        (Custom Protocol)"]
    end

    %% Взаимодействия
    UI <-->|"session/prompt (ACP)"| NB
    NB <--> Engine
    Engine <--> Catalog
    Engine <--> DB
    Engine <--> Scheduler
    
    Scheduler <--> ConnMgr
    ConnMgr <--> ACP_Client
    ConnMgr <--> Codex_Adapt

    ACP_Client <-->|"ACP (с задержкой 1с)"| Gemini
    ACP_Client <-->|"ACP"| Qwen
    Codex_Adapt <-->|"stdio / JSON-RPC"| Codex

    %% Стилизация
    style DB fill:#f9f,stroke:#333,stroke-width:2px
    style Core fill:#bbf,stroke:#333,stroke-width:2px
    style Agents fill:#dfd,stroke:#333,stroke-width:2px
```

## 2. Жизненный цикл выполнения запроса (Turn Lifecycle)

Процесс обработки пользовательского ввода состоит из фазы планирования (выполняется Координатором) и фазы исполнения (выполняется набором Воркеров).

```mermaid
sequenceDiagram
    participant U as Upstream Client
    participant O as OrgeMage
    participant C as Coordinator Agent
    participant W as Worker Agent

    Note over U, O: 1. Инициализация и выбор модели
    U->>O: session/prompt (model: gemini::auto-gemini-3)
    
    Note over O: 2. Фаза планирования (Planning)
    O->>O: Загрузка состояния из SQLite
    O->>C: session/new + initialize
    Note right of C: Ждем 1 сек (стабилизация сессии)
    O->>C: session/prompt (System: "Создай план задач")
    C-->>O: JSON Plan (Task 1, Task 2, Dependencies)
    
    Note over O: 3. Фаза исполнения (Execution)
    O->>O: Построение графа зависимостей
    
    par Выполнение Task 1 (Worker)
        O->>W: session/new
        O->>W: session/prompt (Task 1 details)
        W-->>O: Task Result (Updates, File changes)
    and Выполнение Task 2 (Coordinator as Worker)
        O->>C: session/prompt (Task 2 details)
        C-->>O: Task Result
    end

    Note over O: 4. Нормализация и ответ
    O->>O: Сборка финального ответа
    O-->>U: session/update (Final Summary)
    O->>O: Сохранение состояния в DB
```

## 3. Ключевые технические решения

### 3.1. Нормализация протокола (Protocol Harmonization)
Для обеспечения совместимости с различными реализациями ACP (например, строгий `camelCase` в Gemini и гибкий `snake_case` в Python-агентах), OrgeMage:
*   Дублирует ключевые параметры в запросах: `session_id` (snake) и `sessionId` (camel).
*   Прокидывает обязательные поля, такие как `version` в `clientInfo` и `mcpServers` (даже если список пуст).

### 3.2. Управление состоянием сессий
*   **Задержка стабилизации**: Для CLI-агентов (Gemini) введена пауза в 1000 мс после создания сессии. Это предотвращает ошибки доступа, когда агент еще не успел зафиксировать сессию в своей внутренней БД.
*   **Persistence**: Все сессии, задачи и результаты выполнения сохраняются в SQLite, что позволяет восстанавливать контекст после перезапуска.

### 3.3. Федерация моделей
Каталог моделей динамически собирается со всех подключенных агентов. Каждая модель получает уникальный составной идентификатор `agent_id::model_id`, что позволяет однозначно определять исполнителя задачи.
