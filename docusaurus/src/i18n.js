const translations = {
    'en': {
        'landing.typer': ['easy', 'scalable', 'efficient', 'flexible',]
    },
    'zh-Hans': {
        'landing.typer': ['简单', '可扩展', '高效', '灵活',]
    },
}

export function trans(id, i18nConfig) {
    if (i18nConfig.currentLocale in translations && id in translations[i18nConfig.currentLocale]) {
        return translations[i18nConfig.currentLocale][id]
    } else {
        return translations[i18nConfig.defaultLocale][id]
    }
}