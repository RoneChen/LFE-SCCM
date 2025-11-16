class LFE_SCCM(ScaleHyperprior):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a_ta = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_a_ts = nn.Sequential(
            conv(3, N, kernel_size=5, stride=4),
            GDN(N),
            conv(N, M, stride=4)
        )

        self.g_s_ta = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.g_s_ts = nn.Sequential(
            deconv(M, N, stride=4),
            GDN(N, inverse=True),
            deconv(N, 3, stride=4)
        )

        self.h_a_ta = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_a_ts = nn.Sequential(
            conv(M, N, 3, 1),
            nn.ReLU(),
            conv(N, N)
        )

        self.h_s_ta = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.h_s_ts = nn.Sequential(
            deconv(N, M * 3 // 2, stride=4),
            nn.ReLU(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsampler = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def g_a(self, x, hfreq):
        x_ta = self.g_a_ta(hfreq)
        x_ts = self.g_a_ts(x)
        x =  torch.add(x_ta, x_ts)
        return x

    def h_a(self, x):
        x = torch.abs(x)
        x_ta = self.upsampler(self.h_a_ta(x))
        x_ts = self.h_a_ts(x)
        x = torch.add(x_ta, x_ts)
        return x

    def g_s(self, x):
        x_ta = self.g_s_ta(x)
        x_ts = self.g_s_ts(x)
        x = torch.add(x_ta, x_ts)
        return x

    def h_s(self, x):
        x_ta = self.h_s_ta(x)
        x_ts = self.h_s_ts(x)
        x = self.downsampler(torch.add(x_ta, x_ts))
        return x

    def forward(self, x, hfreq):
        x = x.to(torch.float)
        hfreq = hfreq.to(torch.float)
        y = self.g_a(x, hfreq)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, hfreq):
        y = self.g_a(x, hfreq)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        
        print(f'y, z: {y.shape, z.shape}')
        print(f'y_strings, z_strings: {len(y_strings[0]), len(z_strings[0])}')
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def __str__(self):
        return "LFE_SCCM"