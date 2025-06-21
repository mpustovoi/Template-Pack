#version 450 compatibility
// Расширения GLSL для дополнительных функций
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_shader_image_load_store : enable
#extension GL_ARB_shader_image_size : enable

// Подключение файла с параметрами и константами
#include options.glsl

// ========== КОНСТАНТЫ И НАСТРОЙКИ ==========
#define SceneLights 3         // Количество источников света
#define Pi 3.141592653589     // Число π
#define Epsilon .001          // Малое значение для численной устойчивости

// ========== БУФЕРЫ И ТЕКСТУРЫ ==========
uniform sampler2D depthtex0;                // Буфер глубины
layout(rgba8) uniform image2D colorimg1;    // Воксельная сетка
layout(rgba8) uniform image2D colorimg4;    // Предыдущие кадры (для временного накопления)
uniform sampler2D colortex5;                // Нормали
uniform sampler2D colortex7;                // Временное освещение от блоков
layout(rgba8) uniform image2D colorimg8;    // Текстуры блоков
uniform sampler2D colortex0;                //
const bool colortex4Clear = false;          // Временное накопление

// ========== ПАРАМЕТРЫ ИГРЫ ==========
uniform float frameTimeCounter; // Игровое время
uniform vec3 cameraPosition;    // Позиция камеры
uniform float near, far;        // Плоскости отсечения
uniform vec3 skyColor;          // Цвет неба
uniform float fogDensity;       // Плотность тумана
uniform int isEyeInWater;       // Флаг нахождения в воде
uniform vec3 fogColor;          // Цвет тумана
uniform float sunAngle;         // Угол солнца (время суток)

// ========== СИСТЕМНЫЕ ПЕРЕМЕННЫЕ ==========
uniform vec3 previousCameraPosition;    // Предыдущая позиция камеры
uniform vec3 upPosition;                // Вектор 'вверх' в мировых координатах
uniform mat4 gbufferModelViewInverse;   // Обратная матрица модели-вида
uniform mat4 gbufferPreviousModelView;  // Матрица модели-вида предыдущего кадра  
uniform mat4 gbufferProjectionInverse;  // Обратная матрица проекции
uniform mat4 gbufferPreviousProjection; // Матрица проекции предыдущего кадра
uniform mat4 gbufferProjection;         // Текущая матрица проекции
uniform float viewWidth;                // Ширина viewport
uniform float viewHeight;               // Высота viewport
uniform int heldItemId;                 // ID предмета в основной руке
uniform int heldItemId2;                // ID предмета в второй руке
uniform ivec2 atlasSize;                // Размер атласа текстур
in vec2 texCoord;                       // Текстурные координаты (0-1)
in vec3 vaNormal;                       // Нормаль вершины
vec3 Normal;                            // Нормаль после преобразований
uniform float fogEnd;                   // Конец тумана
uniform float fogStart;                 // Начало тумана
layout(location = 0) out vec4 fragColor;

// ========== ПЕРЕМЕННЫЕ ШЕЙДЕРА ==========
vec3 VoxStart;      // Начальная точка луча
vec3 VoxDir;        // Направление луча
vec3 PlayerView;    // Направление взгляда игрока
vec3 EyeCameraPosition = cameraPosition + gbufferModelViewInverse[3].xyz; // 
int Seed;           // Семя для генератора случайных чисел
int CurrentTri;     // 
int PrevID;         // ID предыдущего вокселя
float depth;        // Глубина текущего пикселя

// Массивы параметров источников света
vec3 LightColor[SceneLights]; // Цвета
vec3 LightPos[SceneLights];   // Позиции
vec2 LightProp[SceneLights];  // Свойства (интенсивность, размер)

// ========== ВОКСЕЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
ivec2 imgSize = imageSize(colorimg1);           // Размер воксельной текстуры
const int BlockDist = 1 << VoxDist;             // Размер воксельной сетки

// ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

// Преобразует координаты с учетом матрицы проекции
vec3 ProjectNDivide(mat4 Matrix, vec3 Pos) {
    vec4 HgnsPos = Matrix * vec4(Pos, 1);
    return HgnsPos.xyz / HgnsPos.w;
}

// Генератор псевдослучайных чисел
float GetRand(in float mi, in float ma) {
    Seed = Seed * 747796405 + 2891336453;
    float hash = fract(sin(float(Seed)) * 43758.5453);
    return mi + (ma - mi) * hash;
}

// Устанавливает направление луча к целевой координате
void PointTowards(in vec3 Coord) {
    VoxDir = normalize(Coord - VoxStart);
}

// ========== ФУНКЦИИ РАБОТЫ С ВОКСЕЛЯМИ ==========

// Проверяет заполненность вокселя
bool GetVoxel(in int ID) {
    ivec2 LoadPos = ivec2((ID % imgSize.x), ID / imgSize.x);
    return (1. - imageLoad(colorimg1, LoadPos).x) != 0.;
}

// Вычисляет нормаль вокселя по его ID
vec3 VoxelNormal(in int ID) {
    vec3 Dir = VoxStart - vec3(
        float(ID % BlockDist) + .5,
        floor(float(ID % (BlockDist * BlockDist)) / float(BlockDist)) + .5,
        floor(float(float(ID) / float(BlockDist * BlockDist))) + .5
    );

    float Inside = ((min(float(abs(Dir.x) > .5 - Epsilon) + float(abs(Dir.y) > .5 - Epsilon) + float(abs(Dir.z) > .5 - Epsilon), float(PrevID != ID))) * 2.) - 1.; // test if inside block
    Inside = 1.;

    if (abs(Dir.x) > max(abs(Dir.y), abs(Dir.z)))
        return vec3(Dir.x / abs(Dir.x), 0, 0) * Inside;
    if (abs(Dir.y) > abs(Dir.z))
        return vec3(0, Dir.y / abs(Dir.y), 0) * Inside;
    return vec3(0, 0, Dir.z / abs(Dir.z)) * Inside;
}

// Получает текстуру и материал вокселя
void GetTexture(in int ID, out vec3 Color, out vec4 Material) {
    int atlastexLocation = int(imageLoad(colorimg1, ivec2(ID%imgSize.x, ID / imgSize.x)).x * 1000.);
    Color = imageLoad(colorimg8, ivec2(mod(VoxStart.xz, 1.) * 16.) + ivec2((atlastexLocation * 16) % atlasSize.x, (atlastexLocation * 16) / atlasSize.x)).rgb;
    Material = vec4(1, 1, 1, 1);
}

// Трассировка через воксельную сетку
float VoxelIntersection(inout int HitID, in float MaxDist) {
    float t = 0.;
    int BlockID;

    vec3 tCoord;
    vec3 Coord = VoxStart;
    float mi;

    // Алгоритм растеризации
    for (int i = 0; i < TraceDist; ++i) {
        tCoord = ((floor(Coord) + vec3(VoxDir.x > 0., VoxDir.y > 0., VoxDir.z > 0.)) - Coord) / VoxDir;
        mi = min(min(tCoord.x, tCoord.y), tCoord.z);
        t += mi + Epsilon;
        if (t > MaxDist) return -1.;
        Coord = VoxStart + (VoxDir * t);

        // Проверка границ воксельной сетки
        if (Coord.x < 0. || Coord.y < 0. || Coord.z < 0.) continue;
        if (Coord.x > float(BlockDist - 1) || Coord.y >= float(BlockDist - 1) || Coord.z >= float(BlockDist - 1)) continue;

        // Проверка заполненности вокселя
        BlockID = int(floor(Coord.x)) + (int(floor(Coord.y)) * BlockDist) + (int(floor(Coord.z)) * BlockDist * BlockDist);
        if (GetVoxel(BlockID) && t > Epsilon) {
            HitID = BlockID;
            return t - Epsilon * 2.;
        }
    }
    return -1.; // Если нет пересечения
}

// Находит ближайшее пересечение и вычисляет цвет освещения
float GetIntersection(inout int HitID, out vec3 LightCol, in float Max) {
    float Closest = Max - Epsilon;
    HitID = -1;
    Closest = VoxelIntersection(HitID, Closest);

    LightCol = vec3(1);
    vec3 TexColor = vec3(1);
    vec4 TexMat = vec4(1, 1, 0, 1);

    if (HitID > -1) GetTexture(HitID, TexColor, TexMat);
    LightCol *= TexColor * (1. - TexMat.z);
    return Closest;
}

// ========== ФУНКЦИИ ИСТОЧНИКОВ СВЕТА ==========

// Инициализация источников света
void LoadLights() {
    float LightAngle = (Pi / 2.) - .1 - (sunAngle * 2. * Pi);
    LightColor[0] = vec3(1, .8, .65);
    LightPos[0] = vec3(sin(LightAngle), cos(LightAngle), .04) * 4500.;
    LightProp[0] = vec2(3000000, 100);

    // Свет луны
    LightColor[1] = vec3(.6, .8, 1); // Луна
    LightPos[1] = -LightPos[0]; // -Солнце
    LightProp[1] = vec2(1000000, 200);

    // Базовые параметры света от руки
    LightColor[2] = vec3(1, 1, 1);
    LightProp[2] = vec2(0, .1414);

    // Позиция света от руки
    LightPos[2] = vec3(BlockDist / 2) + mod(EyeCameraPosition, 1.01);

    // Свет от руки
    int itemId = max(heldItemId, heldItemId2);
    if (itemId > 3999) {
        LightProp[2].x = 1.;

        if (itemId == 4000) LightColor[2] = vec3(1, .7, .4);
        if (itemId == 4001) LightColor[2] = vec3(.4, .7, 1);
        if (itemId == 4002) LightColor[2] = vec3(1, .2, .2);
        if (itemId == 4004) LightColor[2] = vec3(1, .4, 1);
        if (itemId == 4005) LightColor[2] = vec3(.2, 1, .2);
    }
}

// Расчёт освещения от одного источника
vec3 SingleLight(in vec3 VOrigin, in int ID, in vec3 inNormal) {
    // Случайное смещение света
    vec3 Rand = vec3(GetRand(-.5, .5), GetRand(-.5, .5), GetRand(-.5, .5));
    Rand /= sqrt(max(max(pow(Rand.x, 2.), pow(Rand.y, 2.)), pow(Rand.z, 2.)));

    // Направление к источнику света
    VoxDir = (LightPos[ID] + (Rand * LightProp[ID].y)) - VoxStart+vec3(0.25, 0.25, 0.25);
    float Dist = length(VoxDir);
    VoxDir /= Dist;

    // Проверка видимости источника
    float a = dot(VoxDir, inNormal);
    if (a < 0.) {
        VoxStart = VOrigin;
        return vec3(0);
    }

    // Трассировка луча к источнику света
    int HitID = -1;
    vec3 LightCol;
    GetIntersection(HitID, LightCol, Dist);

    VoxStart = VOrigin;
    Dist /= 1.; // Расчёт освещенности по закону обратных квадратов
    return ((LightProp[ID].x * 18. * LightColor[ID] * LightCol * a / max((Dist * Dist), 8.)) / float(LightSamples));
}

// Вычисляет суммарное освещение от всех источников
vec3 GetLight(in vec3 inNormal) {
    vec3 Dir = VoxDir;
    vec3 LightCol;
    for (int i = 0; i < LightSamples / SceneLights; ++i) {
        for (int ID = 0; ID < SceneLights; ++ID) {
            LightCol += SingleLight(VoxStart, ID, inNormal);
        }
    }
    int ID;
    for (int i = 0; i < LightSamples % SceneLights; ++i) {
        ID = int(GetRand(.5, float(SceneLights) + .5) );
        LightCol += SingleLight(VoxStart, ID, inNormal);
    }
    VoxDir = Dir;
    return LightCol;
}

// ========== ФУНКЦИИ ТРАССИРОВКИ ==========

// Изменяет направление луча при отражении/рассеивании
void Redirect(in vec3 inNormal, in vec4 Mat) {
    vec3 Rand = normalize(vec3(GetRand(-1., 1.), GetRand(-1., 1.), GetRand(-1., 1.)));
    if (dot(Rand, inNormal) < 0.)
        Rand = -Rand;
    VoxDir = normalize((reflect(VoxDir, inNormal) * (1. - Mat.x)) + (Rand * Mat.x));
}

// Трассировка пути
vec3 PathTracing() {
    LoadLights();
    PrevID = -1;

    // Инициализация начальной точки и направления
    VoxStart += mat3(gbufferModelViewInverse) *
        ProjectNDivide(gbufferProjectionInverse,
        vec3(texCoord, texture2D(depthtex0, texCoord)) * 2. - 1.) -
        PlayerView * .001;
    vec3 VoxNormal = texture2D(colortex5, texCoord).xyz * 2. - 1.;

    vec3 PixColor = texture(colortex0, texCoord).rgb;
    vec3 VoxColor = texture(colortex0, texCoord).rgb;

    // Первичное освещение
    PixColor *= GetLight(VoxNormal) + texture(colortex7, texCoord).rgb;
    Redirect(VoxNormal, vec4(1, 1, 1, 1));

    // Множественные отражения
    for (int b = 0; b < Redirections; ++b) {
        int HitID = -1;
        vec3 f;
        float t = GetIntersection(HitID, f, float(BlockDist));
        if (HitID == -1)
            return PixColor + (skyColor * VoxColor);
        VoxStart += VoxDir * t;
        vec3 TexColor = fogColor;
        vec4 TexMat = vec4(1, 1, 1, 1);
        if (HitID != -2) {
            VoxNormal = VoxelNormal(HitID);
            GetTexture(HitID, TexColor, TexMat);
        }
        if (b > 0)
            TexColor *= GiStrength;

        VoxColor *= (TexColor * TexMat.z) + vec3(1. - TexMat.z);
        PixColor += VoxColor * GetLight(VoxNormal) * TexMat.x;
        Redirect(VoxNormal, TexMat);
        PrevID = HitID;
    }
    return PixColor;
}

// ========== ОСНОВНАЯ ФУНКЦИЯ ==========
void main() {
    // Инициализация генератора случайных чисел
    Seed = int(mod(float(frameTimeCounter * 1873.), 142.) * 1729. +
    texCoord.x * 1260. + (texCoord.y * 120000.));

    // Расчёт направления взгляда
    vec3 ndc = vec3(texCoord,texture2D(depthtex0, texCoord)) * 2. - 1.;
    PlayerView = normalize(mat3(gbufferModelViewInverse) *
        ProjectNDivide(gbufferProjectionInverse, ndc));

    // Трассировка пути
    vec3 Color;
    depth = (near * far) / (texture2D(depthtex0, texCoord).r * (near - far) + far);

    vec3 Prevndc = vec3(texCoord, texture(depthtex0, texCoord)) * 2. - 1.;
    vec3 PrevViewPos = ProjectNDivide(gbufferProjectionInverse, Prevndc);
    vec3 PrevWorldPos = mat3(gbufferModelViewInverse) * PrevViewPos +
        (cameraPosition - previousCameraPosition);
    vec3 PrevView = (mat3(gbufferPreviousModelView) * PrevWorldPos);
    vec3 ReadCoord = ProjectNDivide(gbufferPreviousProjection, PrevView)* .5 + .5;

    vec4 FrameData = imageLoad(colorimg4, ivec2(ReadCoord.xy * imgSize));

    int passNum = int(FrameData.a * float(FrameAccumulation));
    passNum += 1;

    for (int s = 0; s < RaySamples; ++s) {
        VoxDir = PlayerView;
        VoxStart = vec3(BlockDist / 2) + mod(cameraPosition +
            gbufferModelViewInverse[3].xyz, 1);

        Color += PathTracing();
    }
    Color /= float(RaySamples);

    // Постобработка и вывод
    float Exposure = .5 * Brightness;
    fragColor = vec4((FrameData.rgb * (1. - (1. / float(passNum)))) +
        ((Color * Exposure) / float(passNum)), 1);
    imageStore(colorimg4, ivec2(texCoord * imgSize),
        vec4(fragColor.rgb, clamp(float(passNum + 1)/float(FrameAccumulation), 0., 1.)));
}