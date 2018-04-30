/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <math.h>
#include <assert.h>
#include <stdio.h>

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>


#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_particleRadius(0.125f * 0.5f),
      m_program(0),
      m_vbo(0),
      m_colorVBO(0),
	  m_blurRadius(1.0f),
	  m_depthTex(0),
	  m_bufferSize(256),
	  m_blurProg(0),
	  m_displayTexProg(0),
	  m_renderSurface(false),
	  m_depthFbo(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
	glDeleteTextures(1, &m_depthTex);
	delete m_depthFbo;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);

        if (m_colorVBO)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, m_numParticles);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;

        default:
        case PARTICLE_SPHERES:
            glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

			if (m_renderSurface) {
				calcDepth();
				blurDepth();
				displayTexture();
			}
			else {
				glUseProgram(m_program);
				glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI / 180.0f));
				glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

				glColor3f(1, 1, 1);
				_drawPoints();

				glUseProgram(0);
			}

            
            glDisable(GL_POINT_SPRITE_ARB);
            break;
    }
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);

	if (m_renderSurface) {
		// load shader programs
		m_program = _compileProgram(vertexShader, depthShader);
		m_blurProg = _compileProgram(passThruVS, blurPS);
		m_displayTexProg = _compileProgram(passThruVS, texture2DPS);

		// create depth texture buffer
		m_depthTex = createTexture(GL_TEXTURE_2D, m_bufferSize, m_bufferSize, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		m_depthFbo = new FramebufferObject();
		m_depthFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
		m_depthFbo->IsValid();
	}

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}

GLuint ParticleRenderer::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format)
{
	GLuint texid;
	glGenTextures(1, &texid);
	glBindTexture(target, texid);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
	return texid;
}

void ParticleRenderer::calcDepth()
{
	m_depthFbo->Bind();
	glViewport(0, 0, m_bufferSize, m_bufferSize);
	glUseProgram(m_program);
	bindTexture(m_program, "depthTex", m_depthTex, GL_TEXTURE_2D, 0);
	glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI / 180.0f));
	glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);
	glDisable(GL_DEPTH_TEST);
	glColor3f(1, 1, 1);
	_drawPoints();
	glUseProgram(0);
	m_depthFbo->Disable();
}

void ParticleRenderer::blurDepth()
{
	glUseProgram(m_blurProg);
	bindTexture(m_blurProg, "tex", m_depthTex, GL_TEXTURE_2D, 0);
	glUniform2f(glGetUniformLocation(m_blurProg, "texelSize"), 1.0f / (float)m_bufferSize, 1.0f / (float)m_bufferSize);
	glUniform1f(glGetUniformLocation(m_blurProg, "blurRadius"), m_blurRadius);
	glDisable(GL_DEPTH_TEST);
	drawQuad();
	glUseProgram(0);
}

void ParticleRenderer::displayTexture()
{
	glViewport(0, 0, m_window_w, m_window_h);
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glUseProgram(m_displayTexProg);
	bindTexture(m_displayTexProg, "tex", m_depthTex, GL_TEXTURE_2D, 0);
	drawQuad();
	glUseProgram(0);
	glDisable(GL_BLEND);
}

void ParticleRenderer::bindTexture(GLuint mProg, const char *name, GLuint tex, GLenum target, GLint unit)
{
	GLint loc = glGetUniformLocation(mProg, name);

	if (loc >= 0)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(target, tex);
		glUseProgram(mProg);
		glUniform1i(loc, unit);
		glActiveTexture(GL_TEXTURE0);
	}
	else
	{
#if _DEBUG
		fprintf(stderr, "Error binding texture '%s'\n", name);
#endif
	}
}

void ParticleRenderer::drawQuad()
{
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();
}
