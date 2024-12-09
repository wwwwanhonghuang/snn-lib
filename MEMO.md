1. **Double Exponential Synapse Model**

Let us begin with the double exponential kernel $f(t)$:
$$
f(t) = A[exp(-t/\tau_d)-exp(-t/\tau_r)]
$$
The pre-synapse spike train $S(t) = \sum_{t_k}^t \delta(t-t_k) $​

Post-synapse response $r = f*S$
$$
\frac{d}{dt} ((S*f)(t))=\frac{d}{dt} (\int_{-\infty}^t S(\tau)f(t-\tau)d\tau)=\int_{-\infty}^tS(\tau)\frac{d}{dt}f(t-\tau) d\tau
$$

$$
\frac{d}{dt}f(t-\tau)=\frac{d}{dt} (A[exp(-\frac{t-\tau}{\tau_d})-exp(-\frac{t-\tau}{\tau_r})])=\\
A(\frac{1}{\tau_d}exp(-\frac{t-\tau}{\tau_d})-\frac{1}{\tau_r}exp(-\frac{t-\tau}{\tau_r}))
$$
We now have
$$
\frac{dr}{dt} = \int_{-\infty}^t S(\tau)[\frac{1}{\tau_d}exp(-\frac{t-\tau}{\tau_d})-\frac{1}{\tau_r}exp(-\frac{t-\tau}{\tau_r})]d\tau
$$
Utilize
$$
S(t) = \sum_{t_k}^t \delta(t - t_k)
$$
We get
$$
\frac{dr}{dt} = \sum_{t_k<t}[\frac{1}{\tau_d}exp(-\frac{t-t_k}{\tau_d})-\frac{1}{\tau_r}exp(-\frac{t-t_k}{\tau_r})]
$$
Let auxiliary state $h(t)=\sum_{t_k<t} exp(-\frac{t-t_k}{\tau_r})$
$$
\frac{dr}{dt}=-\frac{r}{\tau_d}+h(t)
$$

$$
\frac{dh}{dt} =-\frac{h}{\tau_r} + \frac{1}{\tau_r\tau_d}\sum_{t_k < t}\delta(t-t_k)
$$

