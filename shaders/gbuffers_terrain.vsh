#version 450 compatibility
#extension GL_ARB_shader_image_load_store : enable
#extension GL_ARB_shader_image_size : enable

#include options.glsl

int VoxelDist = 1<<VoxDist;

out vec2 lmcoord;
out vec2 texcoord;
out vec4 glcolor;
out vec3 Normal;

in vec3 at_midBlock;
uniform int blockEntityId;
in vec3 mc_Entity;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform vec3 cameraPosition;
uniform ivec2 atlasSize;

layout(rgba8) uniform image2D colorimg1;

ivec2 imgSize = imageSize(colorimg1);

void main() {
    gl_Position = ftransform();
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
	glcolor = gl_Color;
    Normal = gl_Normal;

    if(abs(mc_Entity.x-10000.) < 100.) return;
    if(abs(mc_Entity.x-20000.) < 100.) return;
    if(abs(mc_Entity.x-30000.) < 100.) return;
    if(abs(mc_Entity.x-40000.) < 100.) return;

    vec3 EyeCameraPosition = cameraPosition + gbufferModelViewInverse[3].xyz;
    vec3 worldPos = vec3((mat3(gbufferModelViewInverse) * (gl_ModelViewMatrix * gl_Vertex).xyz) + EyeCameraPosition);
        
    ivec3 BlockPos = ivec3(floor(worldPos + (at_midBlock / 64.) + .001));
    ivec3 OffsetBlockPos = BlockPos-ivec3(floor(EyeCameraPosition)) + ivec3(VoxelDist / 2);

    bool canrun;
    if (OffsetBlockPos.x < 0 || OffsetBlockPos.y < 0 || OffsetBlockPos.z < 0) return;
    if (OffsetBlockPos.x >= VoxelDist || OffsetBlockPos.y >= VoxelDist || OffsetBlockPos.z >= VoxelDist) return;

    int ID = (OffsetBlockPos.x + 0) + ((OffsetBlockPos.y + 0) * VoxelDist) + ((OffsetBlockPos.z + 0) * (VoxelDist * VoxelDist));
        
    ivec2 StorePos = ivec2((ID % imgSize.x), ID / imgSize.x);

    ivec2 atlasTex = ivec2(texcoord * vec2(atlasSize/16));
    int atlasIndex = atlasTex.x + (atlasTex.y * atlasSize.x / 16);
    imageStore(colorimg1, StorePos, vec4(float(atlasIndex) / 10000., 0, 0, 0));
}