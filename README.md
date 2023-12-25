# Сервис распознавания штрихкодов

Данный сервис находит на фотографии штрихкод и распознаёт цифры, которые на нём написаны

## Структура репозитория

```
.
├── config - файлы настройки приложения
│
├── src - основной код приложения
│   │
│   ├── containers - контейнеры для DI
│   │
│   ├── routes - настройка эндпоинтов приложения
│   │
│   ├── schemas - классы Pydantic для валидации запросов в routes
│   │
│   └── services - бизнес логика приложения (классы для запуска моделей)
│
├── tests - набор тестов (юнит + интеграционные)
│
└── weights - веса обученной модели
```

## Запуск локально

Чтобы запустить код локально, нужно скачать репозиторий через ```git clone``` а потом выполнить ```make setup``` в корне репозитория

```
git clone ssh://git@gitlab.deepschool.ru:30022/cvr-aug23/m.ivanov/homework2/service.git
make setup
make run_server
```

После этого создастся виртуальное окружение и туда загрузятся зависимости из **requirements.txt**

## Запуск в докере
Для запуска в докере необходимо скачать репозиторий ```git clone``` и запустить контейнер ```make run_docker```

```
git clone ssh://git@gitlab.deepschool.ru:30022/cvr-aug23/m.ivanov/homework2/service.git
sudo make run_docker
```

При таком запуске будет запущена сборка образа, а потом запуск контейнера. Для остановки контейнера выполняем ```sudo make stop_docker```

## Команды make

Для удобного управления репозиторием следует пользоваться скриптами make, но нужно обязательно находится в корне проекта

```
setup - установка репозитория и настройка окружения
check_linter - запуск линтера для проверки качества кода
run_server - запуск сервера локально
run_tests - запуск тестов
run_docker - сборка образа и запуск контейнера
stop_docker - остановка и удаление контейнера
```