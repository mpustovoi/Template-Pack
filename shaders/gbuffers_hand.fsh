#version 450 compatibility

uniform sampler2D lightmap;
uniform sampler2D gtexture;

in vec2 lmcoord;
in vec2 texcoord;
in vec4 glcolor;
in vec3 Normal;

/* DRAWBUFFERS:05 */
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 ScreenNormal;

void main() {
	color = texture(gtexture, texcoord) * glcolor;
	color *= texture(lightmap, lmcoord);
    ScreenNormal = vec4(Normal,1);
	if (color.a < 0.1) {
		discard;
	}
}