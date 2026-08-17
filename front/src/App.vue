<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

type ApiResponse<T> = { status: string; message: string; data: T }
type UserSession = { sessionId: number; userId: number; sessionName: string; createTime: string; updateTime: string }
type ChatRecord = { chatId: number; sessionId: number; userId?: number; question: string; answer: string | null; imageUrls?: string[] }
type ChatTask = { sessionId: number; chatId: number; taskId: string; firstChat: boolean }
type StreamEvent = {
  streamId: string
  taskId: string
  sequence: number
  eventType: 'start' | 'delta' | 'done' | 'error'
  content: string
  errorCode?: string
  errorMessage?: string
}

const token = ref(localStorage.getItem('electricity-token') || '')
const loggedIn = ref(false)
const registerMode = ref(false)
const authLoading = ref(false)
const authError = ref('')
const form = ref({ name: localStorage.getItem('electricity-name') || '', password: '', confirmPassword: '' })

const sessions = ref<UserSession[]>([])
const activeSessionId = ref<number | null>(null)
const messages = ref<ChatRecord[]>([])
const question = ref('')
const sending = ref(false)
const streamStatus = ref('准备就绪')
const pageError = ref('')
const chatContentRef = ref<HTMLElement | null>(null)
let streamController: AbortController | null = null

function authHeaders(): Record<string, string> {
  return { Authorization: `Bearer ${token.value}` }
}

onMounted(async () => {
  if (!token.value) return
  if (!/^[\x20-\x7E]+$/.test(token.value)) {
    logout()
    authError.value = '旧Token包含中文字符，请重新登录'
    return
  }
  try {
    const response = await fetch('/ai/login', { method: 'POST', headers: authHeaders() })
    const data = (await response.json()) as ApiResponse<string>
    if (response.ok && data.status === '200') {
      loggedIn.value = true
      await loadSessions(true)
    } else {
      logout()
    }
  } catch {
    logout()
  }
})

async function submitAuth() {
  authError.value = ''
  const name = form.value.name.trim()
  const password = form.value.password
  if (!name || !password) {
    authError.value = '请输入用户名和密码'
    return
  }
  if (registerMode.value && password !== form.value.confirmPassword) {
    authError.value = '两次输入的密码不一致'
    return
  }

  authLoading.value = true
  try {
    const params = new URLSearchParams({ name, password })
    const endpoint = registerMode.value ? '/ai/register' : '/ai/login'
    const response = await fetch(`${endpoint}?${params}`, { method: 'POST' })
    const data = (await response.json()) as ApiResponse<string>
    if (!response.ok || data.status !== '200' || !data.data) {
      throw new Error(data.message || (registerMode.value ? '注册失败' : '登录失败'))
    }

    token.value = data.data
    loggedIn.value = true
    localStorage.setItem('electricity-token', data.data)
    localStorage.setItem('electricity-name', name)
    form.value.password = ''
    form.value.confirmPassword = ''
    await loadSessions(true)
  } catch (authRequestError) {
    authError.value = authRequestError instanceof Error ? authRequestError.message : '请求失败'
  } finally {
    authLoading.value = false
  }
}

function switchAuthMode() {
  registerMode.value = !registerMode.value
  authError.value = ''
  form.value.password = ''
  form.value.confirmPassword = ''
}

function logout() {
  streamController?.abort()
  token.value = ''
  loggedIn.value = false
  sessions.value = []
  messages.value = []
  activeSessionId.value = null
  localStorage.removeItem('electricity-token')
}

async function loadSessions(openLatest = false) {
  const response = await fetch('/ai/user/sessions', { headers: authHeaders() })
  const data = (await response.json()) as ApiResponse<UserSession[]>
  if (!response.ok || data.status !== '200') throw new Error(data.message || '加载会话失败')
  sessions.value = data.data || []

  if (!openLatest) return

  const latestSession = sessions.value[0]
  if (latestSession) {
    await openSession(latestSession.sessionId)
  } else {
    activeSessionId.value = null
    messages.value = []
  }
}

async function openSession(sessionId: number) {
  if (sending.value) return
  activeSessionId.value = sessionId
  pageError.value = ''
  const response = await fetch(`/ai/user/chat?sessionId=${sessionId}`, { headers: authHeaders() })
  const data = (await response.json()) as ApiResponse<ChatRecord[]>
  if (!response.ok || data.status !== '200') {
    pageError.value = data.message || '加载聊天记录失败'
    return
  }
  messages.value = (data.data || []).map((chat) => ({ ...chat, answer: chat.answer || '' }))
  await loadMessagePlotUrls(messages.value)
  await scrollToBottom()
}

function newConversation() {
  if (sending.value) return
  activeSessionId.value = null
  messages.value = []
  pageError.value = ''
  streamStatus.value = '新对话'
}

async function sendQuestion() {
  const currentQuestion = question.value.trim()
  if (!currentQuestion || sending.value) return

  sending.value = true
  pageError.value = ''
  streamStatus.value = '正在提交任务'
  streamController?.abort()

  const pendingMessage: ChatRecord = {
    chatId: Date.now(),
    sessionId: activeSessionId.value || 0,
    question: currentQuestion,
    answer: '',
    imageUrls: [],
  }
  messages.value.push(pendingMessage)
  question.value = ''
  await scrollToBottom()

  try {
    const response = await fetch('/ai/user/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ sessionId: activeSessionId.value, question: currentQuestion }),
    })
    const data = (await response.json()) as ApiResponse<ChatTask>
    if (!response.ok || data.status !== '200' || !data.data) {
      throw new Error(data.message || '任务提交失败')
    }

    activeSessionId.value = data.data.sessionId
    pendingMessage.sessionId = data.data.sessionId
    pendingMessage.chatId = data.data.chatId
    await loadSessions()
    await readStream(data.data, pendingMessage)
  } catch (sendError) {
    if (sendError instanceof DOMException && sendError.name === 'AbortError') return
    pageError.value = sendError instanceof Error ? sendError.message : '发送失败'
    pendingMessage.answer ||= '回答生成失败，请稍后重试。'
    sending.value = false
  }
}

async function readStream(task: ChatTask, target: ChatRecord) {
  streamController = new AbortController()
  const response = await fetch(`/ai/user/chat/stream?taskId=${encodeURIComponent(task.taskId)}`, {
    headers: { Accept: 'text/event-stream', ...authHeaders() },
    signal: streamController.signal,
  })
  if (!response.ok || !response.body) throw new Error(`SSE连接失败（${response.status}）`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      await handleSseBlock(block, task, target)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
}

async function handleSseBlock(block: string, task: ChatTask, target: ChatRecord) {
  let eventName = 'message'
  const dataLines: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) eventName = line.slice(6).trim()
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (!dataLines.length) return
  if (eventName === 'timeout') throw new Error(dataLines.join('\n'))

  const event = JSON.parse(dataLines.join('\n')) as StreamEvent
  if (event.eventType === 'start') {
    streamStatus.value = '模型正在处理'
  } else if (event.eventType === 'delta') {
    target.answer = (target.answer || '') + (event.content || '')
    streamStatus.value = `正在输出 · ${event.sequence}`
    await scrollToBottom()
  } else if (event.eventType === 'done') {
    target.answer = event.content || target.answer
    streamStatus.value = '回答完成'
    sending.value = false
    target.imageUrls = await loadPlotUrls(task.sessionId, task.chatId)
    await scrollToBottom()
    await loadSessions()
  } else if (event.eventType === 'error') {
    sending.value = false
    throw new Error(event.errorMessage || event.errorCode || 'AI生成失败')
  }
}

async function loadPlotUrls(sessionId: number, chatId: number): Promise<string[]> {
  for (let attempt = 1; attempt <= 5; attempt++) {
    const response = await fetch(`/ai/user/chat/plot?sessionId=${sessionId}&chatId=${chatId}`, { headers: authHeaders() })
    const data = (await response.json()) as ApiResponse<string[]>
    if (response.ok && data.status === '200' && Array.isArray(data.data) && data.data.length) return data.data
    if (attempt < 5) await new Promise((resolve) => window.setTimeout(resolve, 500))
  }
  return []
}

async function loadMessagePlotUrls(records: ChatRecord[]): Promise<void> {
  await Promise.all(
    records.map(async (message) => {
      if (!message.chatId || !message.sessionId) {
        message.imageUrls = []
        return
      }
      message.imageUrls = await loadPlotUrls(message.sessionId, message.chatId)
    }),
  )
}

async function scrollToBottom() {
  await nextTick()
  if (chatContentRef.value) chatContentRef.value.scrollTop = chatContentRef.value.scrollHeight
}

function formatTime(value: string) {
  return value ? value.replace('T', ' ').slice(0, 16) : ''
}

onBeforeUnmount(() => streamController?.abort())
</script>

<template>
  <main v-if="!loggedIn" class="auth-page">
    <section class="brand-panel">
      <div class="brand-mark">⚡</div>
      <p class="eyebrow">ELECTRICITY LLM</p>
      <h1>让电力知识<br />触手可及</h1>
      <p>融合知识检索、智能体分析与流式生成的专业问答平台。</p>
    </section>
    <section class="auth-panel">
      <form class="auth-card" @submit.prevent="submitAuth">
        <p class="eyebrow">WELCOME</p>
        <h2>{{ registerMode ? '创建账号' : '欢迎回来' }}</h2>
        <p class="auth-tip">{{ registerMode ? '注册后将自动登录并进入对话' : '登录后查看历史会话并开始提问' }}</p>
        <label>用户名<input v-model="form.name" autocomplete="username" placeholder="请输入用户名" /></label>
        <label>密码<input v-model="form.password" type="password" autocomplete="current-password" placeholder="请输入密码" /></label>
        <label v-if="registerMode">确认密码<input v-model="form.confirmPassword" type="password" placeholder="请再次输入密码" /></label>
        <p v-if="authError" class="form-error">{{ authError }}</p>
        <button class="primary-button" :disabled="authLoading">{{ authLoading ? '处理中…' : registerMode ? '注册并登录' : '登录' }}</button>
        <button type="button" class="switch-button" @click="switchAuthMode">{{ registerMode ? '已有账号？返回登录' : '没有账号？立即注册' }}</button>
      </form>
    </section>
  </main>

  <main v-else class="chat-layout">
    <aside class="sidebar">
      <div class="sidebar-brand"><span>⚡</span><div><strong>ElectricityLLM</strong><small>电力知识助手</small></div></div>
      <button class="new-chat-button" @click="newConversation">＋ 新对话</button>
      <p class="sidebar-label">历史对话</p>
      <div class="session-list">
        <button v-for="session in sessions" :key="session.sessionId" class="session-item" :class="{ active: activeSessionId === session.sessionId }" @click="openSession(session.sessionId)">
          <strong>{{ session.sessionName || '新对话' }}</strong>
          <small>{{ formatTime(session.updateTime) }}</small>
        </button>
        <p v-if="!sessions.length" class="no-session">还没有历史对话</p>
      </div>
      <div class="account-row"><div class="avatar">{{ form.name.slice(0, 1).toUpperCase() }}</div><span>{{ form.name }}</span><button @click="logout">退出</button></div>
    </aside>

    <section class="conversation">
      <header class="chat-header"><div><strong>{{ sessions.find((item) => item.sessionId === activeSessionId)?.sessionName || '新对话' }}</strong><small>{{ streamStatus }}</small></div><span class="online-dot"></span></header>
      <div ref="chatContentRef" class="message-list">
        <div v-if="!messages.length" class="chat-welcome"><div>⚡</div><h2>你好，我是电力知识助手</h2><p>你可以询问电气工程知识、设备原理，也可以让我根据资料生成图表。</p></div>
        <article v-for="message in messages" :key="message.chatId" class="message-group">
          <div class="bubble-row user"><div class="bubble">{{ message.question }}</div><div class="mini-avatar">你</div></div>
          <div class="bubble-row assistant"><div class="mini-avatar ai">⚡</div><div class="bubble"><span v-if="!message.answer" class="typing">正在思考<span>...</span></span><template v-else>{{ message.answer }}</template><div v-if="message.imageUrls?.length" class="image-grid"><img v-for="url in message.imageUrls" :key="url" :src="url" alt="生成图表" /></div></div></div>
        </article>
        <p v-if="pageError" class="chat-error">{{ pageError }}</p>
      </div>
      <form class="chat-composer" @submit.prevent="sendQuestion"><textarea v-model="question" rows="2" placeholder="输入你的问题，Enter发送，Shift+Enter换行" @keydown.enter.exact.prevent="sendQuestion"></textarea><button :disabled="sending || !question.trim()">{{ sending ? '生成中' : '发送' }}</button></form>
    </section>
  </main>
</template>
