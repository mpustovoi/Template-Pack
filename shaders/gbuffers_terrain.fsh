#version 450 compatibility
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_shader_image_load_store : enable
#extension GL_ARB_shader_image_size : enable

#include options.glsl

uniform sampler2D lightmap;
uniform sampler2D gtexture;
uniform sampler2D normals;
uniform ivec2 atlasSize;
layout(rgba8) uniform image2D colorimg5; // normals
layout(rgba8) uniform image2D colorimg8; // textures

in vec2 lmcoord;
in vec2 texcoord;
in vec4 glcolor;
in vec3 Normal;

ivec2 imgSize = imageSize(colorimg5);

/* DRAWBUFFERS:057 */
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 ScreenNormals;
layout(location = 2) out vec4 Blocklight;

mat3 tbnNormalTangent(vec3 normal, vec3 tangent) {
    vec3 bitangent = cross(tangent, normal);
    return mat3(tangent, bitangent, normal);
}

void main() {
    vec3 ScrNorm = Normal;
    color = texture(gtexture, texcoord) * glcolor;
    ScreenNormals = vec4((Normal + 1.) / 2., 1.);
    Blocklight = vec4(vec3(lmcoord.x), 1.);
    
    if (color.a < .1) {
        discard;
    }
}