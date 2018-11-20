# Соревнование по машинному обучению Telecom Data Cup

Организаторами соревнования являются компании `Mail.Ru Group` и `МегаФон`. 

Официальные ресурсы: [Telecom Data Cup](https://telecomdatacup.ru), платформа проведения [ML Boot Camp](https://mlbootcamp.ru).

## Описание задачи

Было опрошено `9443` абонента. 

Результатом опроса является индекс удовлетворенности для каждого абонента, выраженный нулем (`0` — доволен) и единицей (`1` — не доволен). Необходимо с максимально возможной точностью выявить и предсказать **недовольных** клиентов.

Описание большинства признаков доступно в [файле](https://github.com/MailRuChamps/telecomdatacup/blob/master/!dataset_fields_description.xlsx). Значение некоторых признаков скрыто.

Метрикой задачи является [ROC AUC](http://www.machinelearning.ru/wiki/index.php?title=ROC-кривая). Ответом служит оценка принадлежности к классу, лежащая в диапазоне `[0; 1]` для каждого `SK_ID`.

Предсказание нужно сделать для `5221` абонентов в том же порядке, что и в `subs_csi_test.csv`. Данные опубликованы на сайте [платформы](https://mlbootcamp.ru/round/15/tasks/) (доступны после регистрации). 

Столбцом, с переводом строк (`Line Endings` должно соответствовать формату Windows или Unix). Предварительные результаты будут формироваться по ответам для `2088` абонентов, а финальные по ответам для `3133` абонентов (40/60).

Максимальное количество загрузок решений в день — `5`.

Количество выбираемых решений — `2`.

Разрешено делиться идеями и методиками. Запрещается использование мультиаккаунтов, а также публикация полных бейзлайнов и решений до окончания соревнования. Рекомендуем ознакомиться с [правилами проведения](https://mlbootcamp.ru/media/condition/2018-10-29%20Правила%20конкурса.pdf) Telecom Data Cup.

## Полезная литература

Выражаем свою благодарность преподавателю кафедры Информационные системы и телекоммуникации (МГТУ им. Н.Э. Баумана) — Асмаряну Альберту Эдуардовичу за консультацию и материалы, опубликованные в папке [лекции](https://github.com/MailRuChamps/telecomdatacup/tree/master/lections). 

## Подарки

Участников с лучшими результатами ждут ценные подарки: 
1 место — 400 000 рублей 
2 место — 200 000 рублей 
3 место — 100 000 рублей 
По традиции наших чемпионатов, призеры получат футболки с символикой чемпионата, а именно ТОП200.

## Сообщество ML Boot Camp

Присоединяйтесь к нашему сообществу в Telegram [@mlbootcamp](https://t.me/mlbootcamp). 

Вы всегда можете задать вопросы, получить советы экспертов в области Data Science. Первыми узнать о самых важных граалях. Кроме того, сообщество чемпионатов Mail.Ru Group — это нетворкинг, где легко найти единомышленников. 