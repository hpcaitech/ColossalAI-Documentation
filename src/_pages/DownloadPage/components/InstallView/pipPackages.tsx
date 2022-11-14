import semver from 'semver'
import api from "../../../../core/api"

export async function getPipPkgVersions(url: string): Promise<Set<string>> {
    const response = await api.get(url)
    const text = await response.raw.text()
    const pipPkgVersions: Set<string> = new Set()
    // Python versioning: <public version identifier>[+<local version label>]
    // Public version identifier: N(.N)*[{a|b|rc}N][.postN][.devN]
    // Reference: https://peps.python.org/pep-0440/
    let matched_vers = text.match(/colossalai-\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?%2B.+\.whl/g)
    if (matched_vers) {
        for (let wheel of matched_vers) {
            let version = wheel.split('-')[1].replace('%2B', '+')
            pipPkgVersions.add(version)
        }
    }
    return pipPkgVersions
}

function filterVersions(versions: Set<string>): string[] {
    // return the oldest version and the top-3 latest versions
    let sortedVersions = Array.from(versions)
    sortedVersions.sort((v1, v2) => semver.compare(semver.coerce(v1), semver.coerce(v2)))
    let truncVersions = sortedVersions.slice(-3)
    if (sortedVersions.length > 3) {
        truncVersions.unshift(sortedVersions[0])
    }
    return truncVersions
}

export function getColossalaiVersions(pipPkgVersions: Set<string>): string[] {
    const colossalaiVersions: Set<string> = new Set()
    for (let version of pipPkgVersions) {
        const colossalaiVersion = version.split('+')[0]
        colossalaiVersions.add(colossalaiVersion)
    }
    return filterVersions(colossalaiVersions)
}

export function getTorchVersions(pipPkgVersions: Set<string>): string[] {
    const torchVersions: Set<string> = new Set()
    for (let version of pipPkgVersions) {
        const torchVersion = version.match(/torch([0-9.]+)/)
        torchVersions.add(torchVersion[1])
    }
    return filterVersions(torchVersions)
}

export function getCudaVersions(pipPkgVersions: Set<string>): string[] {
    const cudaVersions: Set<string> = new Set()
    for (let version of pipPkgVersions) {
        const cudaVersion = version.match(/cu([0-9.]+)/)
        cudaVersions.add(cudaVersion[1])
    }
    return filterVersions(cudaVersions)
}
